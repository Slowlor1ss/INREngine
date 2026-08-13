#include "TrainingThreadPool.h"
#include <numeric> // Put this at the very top of your file
#include <random>

#ifdef _PROFILE
#include <nvtx3/nvToolsExt.h>
#endif

#include "GpuDataset.h"
#include "GpuNetwork.h"

// Assuming 'mse' is your mean squared error for the current batch
inline engineFloat CalculatePSNR(engineFloat mse) {
	if (mse <= 0.0000001f) return 100.0f; // Prevent divide by zero (perfect image)
	return -10.0f * std::log10(mse);
}

inline void LogTrainingMetrics(const std::string& csvFilepath, size_t batch, engineFloat cost, engineFloat psnr) 
{
	bool writeHeader = !std::filesystem::exists(csvFilepath);
    
	std::ofstream file(csvFilepath, std::ios::app); // Append mode
	if (file.is_open()) 
	{
		if (writeHeader) 
		{
			file << "Batch,Cost,PSNR\n";
		}
		file << batch << "," << cost << "," << psnr << "\n";
	}
}

//TODO: move
size_t currentEpoch = 0;
size_t maxEpochs = 4000; // From the PyTorch script niters = 2000

static engineFloat RunGradientGuidedTrainingEpoch(Network& network,
                                            const std::vector<ImageUtils::SpatialData>& inputs,
                                            const std::vector<ImageUtils::SpatialData>& targets,
                                            size_t& currentImageIdx,
                                            const size_t printEveryNBatches,
                                            const size_t batchSize,
                                            engineFloat& learningRate,
                                            TrainingThreadPool& threadPool)
{
    engineFloat totalCost = 0.0f;

    for (size_t j = 0; j < printEveryNBatches; j++)
    {
        // Run the entire batch across all cores instantly
        engineFloat batchCost = threadPool.RunBatch(inputs, targets, currentImageIdx, batchSize);
        totalCost += (batchCost / static_cast<engineFloat>(batchSize));
        
        // Apply the averaged batch weights via Adam
        network.ConsumeDelta(learningRate);
        
        // Move forward in the dataset (or pick random)
        //currentImageIdx = GetRandomImageIndex(inputs.size());
    	currentImageIdx = (currentImageIdx + batchSize) % inputs.size();
    	
        // Decay learning rate
		// Comented out for now as were now using Adam
        //learningRate *= static_cast<engineFloat>(std::pow(0.9999999, batchSize));

    	// Calculate the decay factor just like the LambdaLR scheduler
    	engineFloat progress = std::min(static_cast<engineFloat>(currentEpoch) / maxEpochs, 1.0f);
    	//engineFloat progress = min(static_cast<engineFloat>(currentEpoch) / (maxEpochs, 1.0f); // im cheating in 8000 rather then using the 2000 as it just seems to work better TODO; look in to a slower degrading LR
    	// TEST: Decay to 50% (0.5f) of the initial learning rate by the final epoch
    	// instead of a 10% (0.1f)
    	learningRate = config::initial_learning_rate * std::pow(0.5f, progress);

    	currentEpoch++;
    }

    return totalCost / static_cast<engineFloat>(printEveryNBatches);
}

static engineFloat RunGPUTrainingEpoch(
    Network& cpuNetwork, 
    GpuNetwork& gpuNet, 
    GpuDataset& gpuData,
    size_t totalPixels,
    size_t inputChannels,
    size_t targetChannels,
    size_t& currentImageIdx,
    const size_t printEveryNBatches,
    const size_t batchSize,
    engineFloat& learningRate,
    TrainingThreadPool& threadPool,
    const std::vector<ImageUtils::SpatialData>& activeInputs,
    const std::vector<ImageUtils::SpatialData>& activeTargets)
{
    // A hyperparameter to balance how much the network cares about slopes vs colors.
    constexpr engineFloat spatialLossWeight = 0.0f; // Adjust this if you want spatial gradients enabled
	
    for (size_t j = 0; j < printEveryNBatches; j++)
    {
        // Calculate the flat array offsets for this specific batch
        int inOffset = currentImageIdx * inputChannels;
        int tarOffset = currentImageIdx * targetChannels;

    	PROFILE_PUSH_FMT("GPU Batch %zu", currentEpoch);
        // Execute purely on the GPU (No PCIe transfer!)
        gpuNet.TrainBatchGPU(
            gpuData.d_inputAct + inOffset,
            gpuData.d_inputGradX + inOffset,
            gpuData.d_inputGradY + inOffset,
            gpuData.d_targetAct + tarOffset,
            gpuData.d_targetGradX + tarOffset,
            gpuData.d_targetGradY + tarOffset,
            spatialLossWeight,
            learningRate
        );
		PROFILE_POP();
    	
        // Move forward in the dataset
        currentImageIdx = (currentImageIdx + batchSize) % totalPixels;

        // Decay learning rate identically to the CPU version
        engineFloat progress = std::min(static_cast<engineFloat>(currentEpoch) / maxEpochs, 1.0f);
        learningRate = config::initial_learning_rate * std::pow(0.5f, progress);
        currentEpoch++;
    }
	
    // Download parameters to CPU so the Visualizer and Checkpointing work!
    gpuNet.DownloadParametersToCPU(cpuNetwork);
    
    // Quickly run a single batch on the CPU ThreadPool just to calculate the Cost/PSNR metrics for the console
    engineFloat cost = threadPool.RunBatch(activeInputs, activeTargets, currentImageIdx, batchSize) / static_cast<engineFloat>(batchSize);
    return cost;
}

void NeuralImageRecreator()
{
	if(config::use_gpu)
	{
		int deviceCount = 0;
	    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	    std::cout << "\nCUDA devices: " << deviceCount << "\n";
	    for (int i = 0; i < deviceCount; i++)
	    {
	        cudaDeviceProp prop{};
	        cudaGetDeviceProperties(&prop, i);
	        std::cout << "GPU " << i << ": " << prop.name << "\n\n";
	    }
	}

	// Load Input Data & Derive Checkpoint Filename
	const std::string weightsFile = GetCheckpointFilename(config::output_filename, config::output_path);

	BMPParsedData data;
	ParseBMPData(config::target_image_file.c_str(), data);

	// Setup our coordinate mapper lambda for the ImageGenerator
	std::function<ImageUtils::SpatialData(engineFloat, engineFloat)> coordMapper = nullptr;
	if (config::use_positional_encoding) 
	{
	    coordMapper = [freqs = config::pe_num_frequencies](engineFloat x, engineFloat y) {
	        // No longer scaling by 2.0/w! Keep it pure.
	        return PositionalEncodeWithDerivatives(x, y, freqs); 
	    };
	}
	else if (config::use_gaussian_pe)
	{
		GaussianPositionalEncoder gaussianPE(64, 10.0f);
		// TODO: find a better way but
		// we copy gaussianPE into the lambda so it lives forever so dont pass a s reference
		coordMapper = [gaussianPE](engineFloat x, engineFloat y) {
			return gaussianPE(x, y);
		};
	}
	else 
	{
	    coordMapper = [](engineFloat x, engineFloat y) {
	        ImageUtils::SpatialData d;
	        d.values = { x, y };
	        d.gradX = { 1.0f, 0.0f }; // Pure normalized derivative
	        d.gradY = { 0.0f, 1.0f }; // Pure normalized derivative
	        return d;
	    };
	}

	// Generate Target Outputs (RGB + Spatial Edges)
	std::vector<ImageUtils::SpatialData> targetSpatialData = ImageUtils::GenerateGradientTargets(data.outputs, data.width, data.height);

	// Generate Inputs (Coordinates/PE + Spatial Slopes)
	std::vector<ImageUtils::SpatialData> inputSpatialData;
	inputSpatialData.reserve(data.inputs.size());
	for (const auto& rawInput : data.inputs) {
		inputSpatialData.push_back(coordMapper(rawInput[0], rawInput[1]));
	}

	///
	// TODO: resize vectors in if statement to not wase memory do this soon or will forget
	// Create new vectors to hold the randomized data
	std::vector<ImageUtils::SpatialData> shuffledInputs(inputSpatialData.size());
	std::vector<ImageUtils::SpatialData> shuffledTargets(targetSpatialData.size());
	if( config::shuffle_pixel_batch )
	{
		std::cout << "Shuffling dataset for random pixel batching...\n";

		// Create an array of indices [0, 1, 2, ... 65535]
		std::vector<size_t> indices(inputSpatialData.size());
		std::iota(indices.begin(), indices.end(), 0);

		// Shuffle the indices
		std::mt19937 g(42); // Fixed seed for consistency
		std::shuffle(indices.begin(), indices.end(), g);


		for(size_t i = 0; i < indices.size(); ++i) {
		    shuffledInputs[i] = inputSpatialData[indices[i]];
		    shuffledTargets[i] = targetSpatialData[indices[i]];
		}
	}
	///

	// Initialize Neural Network & Visualizer Window
	//size_t inputLayerSize = config::use_positional_encoding ? (config::pe_num_frequencies * 4) : 2;
	// Run a dummy coordinate through the mapper to see how big the output is (depends on what mode we run in)
	ImageUtils::SpatialData dummyData = coordMapper(0.0f, 0.0f);
	size_t inputLayerSize = dummyData.values.size();
	
	std::vector<size_t> layerDims;
	layerDims.push_back(inputLayerSize);
	if (!config::custom_layer_dims.empty()) {
		layerDims.insert(layerDims.end(), config::custom_layer_dims.begin(), config::custom_layer_dims.end());
	} else {
		//layerDims.insert(layerDims.end(), { 64, 64, 64, 64, 3 });
		layerDims.insert(layerDims.end(), { 256, 256, 3 });

	}
	
	// std::vector<ActFunc::Base*> activations = {
	// 	ActFunc::DataBase::FindActFunc<ActFunc::Wire>(),   // Layer 1: The Localized Denoiser
	// 	ActFunc::DataBase::FindActFunc<ActFunc::Siren>(),  // Layer 2: The High-Frequency Fitter
	// 	ActFunc::DataBase::FindActFunc<ActFunc::None>()    // Layer 3: Output Linear Projection
	// };
	
	// Pass the activation functions dynamically!
	Network network{
		layerDims,
		config::custom_activations,
		CostFunc::DataBase::FindCostFunc(CostFunc::Charbonnier::k_name)
		//CostFunc::DataBase::FindCostFunc(CostFunc::MSE::k_name)
		
		//ActFunc::DataBase::FindActFunc<ActFunc::Wire>(),
		// TODO: look in to this more maybe just use sigmoid as its basically the same or none as its more truthfully ig
		// and the docmentation says to just use a linear or sine https://deepwiki.com/vsitzmann/siren/2-siren-architecture#sinelayer-and-network-structure
		//ActFunc::DataBase::FindActFunc<ActFunc::Sigmoid>() 
	};

	// Create the thread pool using your Ryzen's hardware concurrency (usually 16 threads for the 4800H)
	unsigned int numThreads = std::thread::hardware_concurrency();
	if (numThreads == 0) numThreads = 8;
	TrainingThreadPool threadPool(numThreads, network);

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	std::unique_ptr<GpuNetwork> gpuNet = nullptr;
	std::unique_ptr<GpuDataset> gpuData = nullptr;

	const auto& activeInputs = config::shuffle_pixel_batch ? shuffledInputs : inputSpatialData;
	const auto& activeTargets = config::shuffle_pixel_batch ? shuffledTargets : targetSpatialData;
	const size_t totalPixels = activeInputs.size();
	const size_t inChan = inputLayerSize;
	const size_t tarChan = activeTargets.empty() ? 3 : activeTargets[0].values.size();

	// Calculate Padded Dataset Size (must be a multiple of batch_size)
	const size_t numBatches = (totalPixels + config::batch_size - 1) / config::batch_size;
	const size_t paddedPixels = numBatches * config::batch_size;
	
	if (config::use_gpu)
	{
		std::cout << "Initializing GPU Pipeline...\n";
		gpuNet = std::make_unique<GpuNetwork>(network, config::batch_size);
		gpuData = std::make_unique<GpuDataset>(paddedPixels, inChan, tarChan);
		
		std::cout << "Flattening dataset for VRAM transfer...\n";
		std::vector<engineFloat> flatInAct(paddedPixels * inChan), flatInGradX(paddedPixels * inChan), flatInGradY(paddedPixels * inChan);
		std::vector<engineFloat> flatTarAct(paddedPixels * tarChan), flatTarGradX(paddedPixels * tarChan), flatTarGradY(paddedPixels * tarChan);
		
		for (size_t i = 0; i < paddedPixels; ++i) {
			// Wrap around to the start of the image if we need extra pixels to fill the final batch
			size_t srcIdx = i % totalPixels;
			
			for (size_t c = 0; c < inChan; ++c) {
				flatInAct[i * inChan + c] = activeInputs[srcIdx].values[c];
				flatInGradX[i * inChan + c] = activeInputs[srcIdx].gradX[c];
				flatInGradY[i * inChan + c] = activeInputs[srcIdx].gradY[c];
			}
			for (size_t c = 0; c < tarChan; ++c) {
				flatTarAct[i * tarChan + c] = activeTargets[srcIdx].values[c];
				flatTarGradX[i * tarChan + c] = activeTargets[srcIdx].gradX[c];
				flatTarGradY[i * tarChan + c] = activeTargets[srcIdx].gradY[c];
			}
		}
		
		gpuData->UploadData(flatInAct, flatInGradX, flatInGradY, flatTarAct, flatTarGradX, flatTarGradY);
		std::cout << "GPU Dataset Uploaded.\n";
	}
	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	
	ImageWindow rendererWindow(int(data.width * config::output_image_scale), int(data.height * config::output_image_scale));

	// Load Weights Checkpoint (Validates dimensions vs current layerDims automatically)
	// TODO: REENABLE ONCE I FIXED THIS DAMMED BUG
	if (!config::benchmark_enabled)
	{
		LoadCheckpoint(network, weightsFile);
		// If we loaded CPU weights from disk, we need to push them into the GPU
		if (config::use_gpu) { 
			// TODO: maybe do this better
			// We can simply destroy and recreate the gpuNet to pull the new weights
			gpuNet = std::make_unique<GpuNetwork>(network, config::batch_size);
		}
	}
	
	// Display Interactive Controls
	PrintControls();

	size_t currentImage = 0;
	bool liveUpdateWindow = config::initial_live_update_state;
	engineFloat learningRate = config::initial_learning_rate;

	// Main Training & UI Loop
	while (true)
	{
		// Pump Windows messages so the viewer window stays responsive
		rendererWindow.ProcessMessages();

		// Handle non-blocking user input
		if (!HandleUserAction(PollUserAction(), network, data, weightsFile, liveUpdateWindow, coordMapper, threadPool))
		{
			break;
		}

		// Perform batch training step
		engineFloat cost;
		if (config::use_gpu) 
		{
			cost = RunGPUTrainingEpoch(
				network, *gpuNet, *gpuData, paddedPixels, inChan, tarChan, 
				currentImage, config::print_every_n_batches, config::batch_size, 
				learningRate, threadPool, activeInputs, activeTargets
			);
		}
		else
		{
			if( config::shuffle_pixel_batch )
				cost = RunGradientGuidedTrainingEpoch(network, shuffledInputs, shuffledTargets, currentImage, config::print_every_n_batches, config::batch_size, learningRate, threadPool);
			else
				cost = RunGradientGuidedTrainingEpoch(network, inputSpatialData, targetSpatialData, currentImage, config::print_every_n_batches, config::batch_size, learningRate, threadPool);
		}

		engineFloat currentPSNR = CalculatePSNR(cost);
		
		if (config::benchmark_enabled)
		{
			// Ensure the output directories exist
			std::string stem = fs::path(config::output_filename).stem().string();
			std::filesystem::path basePath(config::output_path);
			auto benchFolder = basePath / ("benchmark_" + stem);
			std::filesystem::create_directories(benchFolder);
			std::filesystem::create_directories(benchFolder / "rgb");
			std::filesystem::create_directories(benchFolder / "grad");

			// Log Metrics to CSV
			std::string csvPath = (benchFolder / "metrics.csv").string();
			LogTrainingMetrics(csvPath, currentEpoch, cost, currentPSNR);

			// Save Image Frames
			auto rgbImage = GenerateReconstructedImage(network, data.width*config::output_image_scale, data.height*config::output_image_scale, coordMapper, RenderMode::StandardRGB, threadPool);
			std::string rgbPath = std::format("{}/rgb/step_{:05d}.bmp", benchFolder.string(), currentEpoch);
			saveBMP(rgbPath, data.width*config::output_image_scale, data.height*config::output_image_scale, rgbImage);

			auto gradImage = GenerateReconstructedImage(network, data.width*config::output_image_scale, data.height*config::output_image_scale, coordMapper, RenderMode::SpatialGradient, threadPool);
			std::string gradPath = std::format("{}/grad/step_{:05d}.bmp", benchFolder.string(), currentEpoch);
			saveBMP(gradPath, data.width*config::output_image_scale, data.height*config::output_image_scale, gradImage);
		}
		
		// Live viewer update
		if (liveUpdateWindow)
		{
			rendererWindow.Update(GenerateReconstructedImage(network, int(data.width * config::output_image_scale), int(data.height * config::output_image_scale), coordMapper, config::render_mode, threadPool));
		}

		// Report progress
		std::cout << "COST: " << cost << " LR: " << learningRate << " PSNR(dB): " << currentPSNR << '\n';
		
		if (config::benchmark_enabled && currentEpoch >= maxEpochs)
		{
			std::cout << "Max epochs reached! Auto-quitting for benchmark pipeline...\n";
			break;
		}
	}

	// Final Output Generation & Cleanup
	std::cout << "Generating final output image from network state...\n";

	std::vector<engineFloat> finalReconstructedImage = GenerateReconstructedImage(network, int(data.width * config::output_image_scale), int(data.height * config::output_image_scale), coordMapper, config::render_mode, threadPool);

	rendererWindow.Update(finalReconstructedImage);

	if (!config::benchmark_enabled)
	{
		saveBMP("network_output.bmp", data.width, data.height, finalReconstructedImage);
		std::cout << "Successfully saved network_output.bmp!\n";
	}

	if (config::benchmark_enabled)
	{
		// Ensure the output directories exist
		std::string stem = fs::path(config::output_filename).stem().string();
		std::filesystem::path basePath(config::output_path);
		auto benchFolder = basePath / ("benchmark_" + stem);
		const std::string benchWeightsFile = GetCheckpointFilename(config::output_filename, benchFolder.string());
		SaveCheckpoint(network, benchWeightsFile);
		std::cout << "Final " << benchWeightsFile << " saved.\n";
	}
	else
	{
		SaveCheckpoint(network, weightsFile);
		std::cout << "Final " << weightsFile << " saved.\n";
	}
}