#include "NeuralImageRecreator.h"
#include "Config.h"
#include "CheckpointUtils.h"
#include "TrainingUserActions.h"
#include "TrainingMetricsReporter.h"
#include "CostFuncDataBase.h"
#include "ImageGenerator.h"
#include "GridEncoding.cuh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

NeuralImageRecreator::InitData NeuralImageRecreator::LoadInitData()
{
	InitData init;

	init.weightsFile = GetCheckpointFilename(config::output_filename, config::output_path);

	ParseBMPData(config::target_image_file.c_str(), init.data);

	// Create a flattened version of the target image for full-reference metrics
	init.flatTargetImage = FlattenTargetImage(init.data);

	// Setup our coordinate mapper lambda for the ImageGenerator
	init.coordMapper = BuildCoordMapper();

	init.dataset = BuildSpatialDataset(init.data, init.coordMapper);

	// Grid encoder replaces the coordMapper as Layer 0's input when enabled ; Layer 0
	// must match the encoder's concatenated output width, not coordMapper's
	init.inputLayerSize = config::use_grid_encoding
		? static_cast<size_t>(GridEncoding::Config::NumLevels * GridEncoding::Config::FeaturesPerLevel)
		: init.coordMapper(0.0f, 0.0f).values.size();

	return init;
}

std::vector<size_t> NeuralImageRecreator::BuildLayerDims(size_t inputLayerSize)
{
	std::vector<size_t> layerDims;
	layerDims.push_back(inputLayerSize);
	if (!config::custom_layer_dims.empty()) {
		layerDims.insert(layerDims.end(), config::custom_layer_dims.begin(), config::custom_layer_dims.end());
	} else {
		// TODO: we should proabably remove this as we basically always ste this in config anyway, but maybe keep just in case
		// just know this will basically never be hit
		//layerDims.insert(layerDims.end(), { 64, 64, 64, 64, 3 });
		layerDims.insert(layerDims.end(), { 256, 256, 3 });
	}
	return layerDims;
}

NeuralImageRecreator::NeuralImageRecreator()
	: m_init(LoadInitData())
	, m_network(BuildLayerDims(m_init.inputLayerSize)
	, config::custom_activations
	, CostFunc::DataBase::FindCostFunc(CostFunc::Charbonnier::k_name))
	, m_threadPool(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 8
	, m_network)
{
	m_inChan = m_init.inputLayerSize;
	m_tarChan = m_init.dataset.targets.empty() ? 3 : m_init.dataset.targets[0].values.size();

	m_renderWidth = m_init.data.width * static_cast<int>(config::output_image_scale);
	m_renderHeight = m_init.data.height * static_cast<int>(config::output_image_scale);

	// Load checkpoint into the CPU network FIRST ; *Gpu*Network's constructor copies weights
	// from the CPU network at construction time only, so it must happen before GpuTrainingPipeline
	// is constructed below or the loaded weights never make it to VRAM :)
	if (!config::benchmark_enabled) // We dont load with when benchmarking as this could mess up our data
	{
		LoadCheckpoint(m_network, m_init.weightsFile);
	}

	if (config::use_gpu)
	{
		m_gpu = std::make_unique<GpuTrainingPipeline>(m_network, m_init.dataset, m_inChan, m_tarChan,
		                                              m_renderWidth, m_renderHeight, m_init.coordMapper);
	}

	m_window = std::make_unique<ImageWindow>(m_renderWidth, m_renderHeight);

	m_learningRate = config::initial_learning_rate;
	m_liveUpdateWindow = config::initial_live_update_state;
}

engineFloat NeuralImageRecreator::RunCpuEpoch(size_t& currentImageIdx, size_t printEveryNBatches, size_t batchSize, engineFloat& learningRate)
{
	engineFloat totalCost = 0.0f;

	for (size_t j = 0; j < printEveryNBatches; j++)
	{
		// Run the entire batch across all cores instantly
		engineFloat batchCost = m_threadPool.RunBatch(m_init.dataset.inputs, m_init.dataset.targets, currentImageIdx, batchSize);
		totalCost += (batchCost / static_cast<engineFloat>(batchSize));

		// Apply the averaged batch weights via Adam
		m_network.ConsumeDelta(learningRate);

		// Move forward in the dataset (or pick random)
		currentImageIdx = (currentImageIdx + batchSize) % m_init.dataset.inputs.size();

		// Calculate the decay factor just like the LambdaLR scheduler
		engineFloat progress = std::min(static_cast<engineFloat>(m_currentEpoch) / config::max_epochs, 1.0f);
		// TEST: Decay to 50% (0.5f) of the initial learning rate by the final epoch
		// instead of a 10% (0.1f)
		learningRate = config::initial_learning_rate * std::pow(0.5f, progress);

		m_currentEpoch++;
	}

	return totalCost / static_cast<engineFloat>(printEveryNBatches);
}

std::vector<engineFloat> NeuralImageRecreator::RenderLiveFrame(int totalRenderPixels, size_t tarChan)
{
	if (config::use_gpu)
	{
		// TODO: support rendering spatial gradient
		return m_gpu->RenderFrame(totalRenderPixels, tarChan);
	}

	// Fallback if not using gpu
	return GenerateReconstructedImage(m_network, m_renderWidth, m_renderHeight, m_init.coordMapper, config::render_mode, m_threadPool);
}

void NeuralImageRecreator::SaveFinalOutputs(GpuNetwork* gpuNet)
{
	std::cout << "Generating final output image from network state...\n";

	std::vector<engineFloat> finalReconstructedImage = GenerateReconstructedImage(
		m_network, m_renderWidth, m_renderHeight, m_init.coordMapper, config::render_mode, m_threadPool);

	m_window->Update(finalReconstructedImage);

	if (!config::benchmark_enabled)
	{
		saveBMP("network_output.bmp", m_init.data.width, m_init.data.height, finalReconstructedImage);
		std::cout << "Successfully saved network_output.bmp!\n";
	}

	if (config::benchmark_enabled)
	{
		// Ensure the output directories exist
		std::string stem = fs::path(config::output_filename).stem().string();
		fs::path basePath(config::output_path);
		auto benchFolder = basePath / ("benchmark_" + stem);
		const std::string benchWeightsFile = GetCheckpointFilename(config::output_filename, benchFolder.string());
		SaveCheckpoint(m_network, gpuNet, benchWeightsFile);
		std::cout << "Final " << benchWeightsFile << " saved.\n";
	}
	else
	{
		SaveCheckpoint(m_network, gpuNet, m_init.weightsFile);
		std::cout << "Final " << m_init.weightsFile << " saved.\n";
	}
}

void NeuralImageRecreator::Run()
{
	// TODO: Maybe move this in main but then we do poll user action in here so...
	// Display Interactive Controls
	PrintControls();
	
	// In case we do not use the gpu version we need to make sure we dont dereference a nullptr
	GpuNetwork* gpuNet = m_gpu ? &m_gpu->GetNetwork() : nullptr;

	// Main Training & UI Loop
	while (true)
	{
		// Pump Windows messages so the viewer window stays responsive
		m_window->ProcessMessages();

		// Handle non-blocking user input
		if (!HandleUserAction(PollUserAction(), m_network, gpuNet, m_init.data, m_init.weightsFile,
		                      m_liveUpdateWindow, m_init.coordMapper, m_threadPool))
		{
			break;
		}

		// Perform batch training step
		engineFloat cost;
		if (config::use_gpu)
		{
			cost = m_gpu->RunEpoch(m_currentImage, config::print_every_n_batches, config::batch_size, m_learningRate, m_currentEpoch);
		}
		else
		{
			cost = RunCpuEpoch(m_currentImage, config::print_every_n_batches, config::batch_size, m_learningRate);
		}

		if (config::benchmark_enabled)
		{
			RunBenchmarkStep(m_network, m_init.coordMapper, m_threadPool, m_init.data, m_init.flatTargetImage, m_currentEpoch, cost);
		}

		// Live viewer update
		if (m_liveUpdateWindow)
		{
			std::vector<engineFloat> rgbImage = RenderLiveFrame(m_renderWidth * m_renderHeight, m_tarChan);
			m_window->Update(rgbImage);

			ReportProgress(cost, m_learningRate, rgbImage, m_init.flatTargetImage);

		}
		
		if (config::benchmark_enabled && m_currentEpoch >= config::max_epochs)
		{
			std::cout << "Max epochs reached! Auto-quitting for benchmark pipeline...\n";
			break;
		}
	}

	// Final Output Generation & Cleanup
	SaveFinalOutputs(gpuNet);
}