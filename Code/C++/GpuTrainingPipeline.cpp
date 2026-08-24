#include "GpuTrainingPipeline.h"
#include "Config.h"
#include "TrainingDataset.h"

#include "Network.h"
#include "GpuNetwork.h"
#include "GpuDataset.h"
#include "../cuda/GridEncoding.cuh"
#include "../cuda/CudaManager.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef _PROFILE
#include <nvtx3/nvToolsExt.h>
#endif

namespace
{
	// Prints out the CUDA devices available on this pc
	void PrintCudaDeviceInfo()
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
}

GpuTrainingPipeline::GpuTrainingPipeline(Network& cpuNetwork, const SpatialDataset& dataset, size_t inChan, size_t tarChan,
                                         int renderWidth, int renderHeight,
                                         int sourceWidth, int sourceHeight,
                                         const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordMapper)
	: m_inChan(inChan), m_tarChan(tarChan), m_totalPixels(dataset.inputs.size()), m_sourceWidth(sourceWidth), m_sourceHeight(sourceHeight)
{
	PrintCudaDeviceInfo();

	// Calculate Padded Dataset Size (must be a multiple of batch_size)
	const size_t numBatches = (m_totalPixels + config::batch_size - 1) / config::batch_size;
	m_paddedPixels = numBatches * config::batch_size;

	std::cout << "Initializing GPU Pipeline...\n";
	m_net = std::make_unique<GpuNetwork>(cpuNetwork, config::batch_size, config::use_grid_encoding);
	m_data = std::make_unique<GpuDataset>(m_paddedPixels, inChan, tarChan);

	std::cout << "Flattening dataset for VRAM transfer...\n";
	FlattenAndUploadDataset(inChan, tarChan, dataset);
	std::cout << "GPU Dataset Uploaded.\n";
	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	// Render buffer setup, used for the live viewer + final/export frames
	int totalRenderPixels = renderWidth * renderHeight;

	CUDA_CHECK(cudaMalloc(&d_renderColors, totalRenderPixels * tarChan * sizeof(engineFloat)));

	if (config::use_grid_encoding)
	{
		CUDA_CHECK(cudaMalloc(&d_renderPixelX, totalRenderPixels * sizeof(engineFloat)));
		CUDA_CHECK(cudaMalloc(&d_renderPixelY, totalRenderPixels * sizeof(engineFloat)));

		std::vector<engineFloat> h_renderPixelX(totalRenderPixels);
		std::vector<engineFloat> h_renderPixelY(totalRenderPixels);

		for (int y = 0; y < renderHeight; ++y)
		{
			for (int x = 0; x < renderWidth; ++x)
			{
				int idx = y * renderWidth + x;
				// Pass [-1.0, 1.0] so FindCell's (x + 1.0f) * 0.5f maps correctly to [0.0, 1.0]
				h_renderPixelX[idx] = (static_cast<engineFloat>(x) / static_cast<engineFloat>(renderWidth)) * 2.0f - 1.0f;
				h_renderPixelY[idx] = (static_cast<engineFloat>(y) / static_cast<engineFloat>(renderHeight)) * 2.0f - 1.0f;
			}
		}

		CUDA_CHECK(cudaMemcpy(d_renderPixelX, h_renderPixelX.data(), totalRenderPixels * sizeof(engineFloat), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_renderPixelY, h_renderPixelY.data(), totalRenderPixels * sizeof(engineFloat), cudaMemcpyHostToDevice));
	}
	else
	{
		// Fallback buffer so standard Positional Encoding doesn't crash the GPU
		CUDA_CHECK(cudaMalloc(&d_renderInputs, totalRenderPixels * inChan * sizeof(engineFloat)));
		std::vector<engineFloat> h_renderInputs(totalRenderPixels * inChan);

		for (int y = 0; y < renderHeight; ++y)
		{
			for (int x = 0; x < renderWidth; ++x)
			{
				int pixelIdx = (y * renderWidth + x) * inChan;
				engineFloat normX = (static_cast<engineFloat>(x) / static_cast<engineFloat>(renderWidth)) * 2.0f - 1.0f;
				engineFloat normY = (static_cast<engineFloat>(y) / static_cast<engineFloat>(renderHeight)) * 2.0f - 1.0f;

				ImageUtils::SpatialData encoded = coordMapper(normX, normY);
				for (size_t c = 0; c < encoded.values.size(); ++c) {
					h_renderInputs[pixelIdx + c] = encoded.values[c];
				}
			}
		}
		CUDA_CHECK(cudaMemcpy(d_renderInputs, h_renderInputs.data(), h_renderInputs.size() * sizeof(engineFloat), cudaMemcpyHostToDevice));
	}
	std::cout << "Finished GPU Render setup.\n";
}

GpuTrainingPipeline::~GpuTrainingPipeline()
{
	if (d_renderPixelX) cudaFree(d_renderPixelX);
	if (d_renderPixelY) cudaFree(d_renderPixelY);
	if (d_renderInputs) cudaFree(d_renderInputs);
	if (d_renderColors) cudaFree(d_renderColors);
}

engineFloat GpuTrainingPipeline::RunEpoch(size_t& currentImageIdx, size_t printEveryNBatches, size_t batchSize,
                                          engineFloat& learningRate, size_t& currentEpoch)
{
	for (size_t j = 0; j < printEveryNBatches; j++)
	{
		// Calculate the flat array offsets for this specific batch
		int inOffset = static_cast<int>(currentImageIdx * m_inChan);
		int tarOffset = static_cast<int>(currentImageIdx * m_tarChan);

		engineFloat jitterAmpX = config::denoise_jitter_strength * (2.0f / static_cast<engineFloat>(m_sourceWidth));
		engineFloat jitterAmpY = config::denoise_jitter_strength * (2.0f / static_cast<engineFloat>(m_sourceHeight));
		
		PROFILE_PUSH_FMT("GPU Batch %zu", currentEpoch);
		// Execute purely on the GPU (No PCIe transfer!)
		m_net->TrainBatchGPU(
			m_data->d_inputAct + inOffset,
			m_data->d_inputGradX + inOffset,
			m_data->d_inputGradY + inOffset,
			m_data->d_targetAct + tarOffset,
			m_data->d_targetGradX + tarOffset,
			m_data->d_targetGradY + tarOffset,
			m_data->d_pixelX + currentImageIdx,
			m_data->d_pixelY + currentImageIdx,
			config::spatialLossWeight,
			learningRate,
			config::use_denoise_jitter,
			jitterAmpX,
			jitterAmpY
		);
		PROFILE_POP();

		// Move forward in the dataset
		//currentImageIdx = (currentImageIdx + batchSize) % m_totalPixels;
		
		// TODO: we need to fix this as our batchsize is not divisable by the total pixes (often at least)
		// we need to wrap around or do something; Maybe we can allocate a d_pixelIndices buffer containing 
		// numbers 0 to m_totalPixels, shuffle it randomly every epoch, and have the GPU pull random pixels for every batch
		//
		// For now well just move back a bit e.g. if batchsize=30 totalPx=100 we do 0-29, 30-59, 60-89, (move back) 70-99
		currentImageIdx += batchSize;
		if (currentImageIdx >= m_totalPixels) 
			currentImageIdx = 0; 
		else if (currentImageIdx + batchSize > m_totalPixels) 
			currentImageIdx = m_totalPixels - batchSize;
		
		// Decay learning rate identically to the CPU version
		engineFloat progress = std::min(static_cast<engineFloat>(currentEpoch) / config::max_epochs, 1.0f);
		learningRate = config::initial_learning_rate * std::pow(0.5f, progress);
		currentEpoch++;
	}

	// Step back to the target offset we just trained on
	size_t lastImageIdx = (currentImageIdx + m_totalPixels - batchSize) % m_totalPixels;
	int lastTarOffset = static_cast<int>(lastImageIdx * m_tarChan);
	engineFloat batchCost = m_net->GetLastBatchCost(m_data->d_targetAct + lastTarOffset);
	return batchCost;
}

std::vector<engineFloat> GpuTrainingPipeline::RenderFrame(int totalRenderPixels, size_t tarChan) const
{
	// Run inference completely on the GPU using the chunked PredictGPU
	m_net->PredictGPU(d_renderPixelX, d_renderPixelY, d_renderInputs, d_renderColors, totalRenderPixels);

	// Download only the final image colors
	std::vector<engineFloat> h_colors(totalRenderPixels * tarChan);
	CUDA_CHECK(cudaMemcpy(h_colors.data(), d_renderColors, h_colors.size() * sizeof(engineFloat), cudaMemcpyDeviceToHost));

	std::vector<engineFloat> rgbImage;
	if (tarChan == 3)
	{
		rgbImage = h_colors;
	}
	else if (tarChan == 1)
	{
		rgbImage.resize(totalRenderPixels * 3);
		for (int i = 0; i < totalRenderPixels; ++i) {
			rgbImage[i * 3 + 0] = h_colors[i];
			rgbImage[i * 3 + 1] = h_colors[i];
			rgbImage[i * 3 + 2] = h_colors[i];
		}
	}
	return rgbImage;
}

#ifdef _WIN32
#include <windows.h> // For checking host ram size
#endif

// +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
//	Note: The flattening of big images runs increadibly slow on my old laptop, after some profiling I found out that it's due to having a low about of ram
//		  I made a seperate funtion thats optimized for memory useage, which makes rendering big images actually feasble on my laptop, but the code is a bit ugly
//		  and therefore banished to this bottom section, by all means this is horrible.
// +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
void GpuTrainingPipeline::FlattenAndUploadDataset(size_t inChan, size_t tarChan, const SpatialDataset& dataset)
{
	unsigned long long totalRamGB = 0;
	// Get sys RAM
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        totalRamGB = memInfo.ullTotalPhys / (1024ull * 1024ull * 1024ull);
    }
#endif

    std::cout << "Detected System RAM: " << totalRamGB << " GB.\n";

    if (totalRamGB > 16) {
        std::cout << "Using bulk dataset upload...\n";
        FlattenAndUploadDatasetBluk(inChan, tarChan, dataset);
    } else {
        std::cout << "Using chunked dataset upload (due to low RAM :D)...\n";
        FlattenAndUploadDatasetChunked(inChan, tarChan, dataset);
    }
}

void GpuTrainingPipeline::FlattenAndUploadDatasetBluk(size_t inChan, size_t tarChan, const SpatialDataset& dataset)
{
	std::vector<engineFloat> flatInAct(m_paddedPixels * inChan), flatInGradX(m_paddedPixels * inChan), flatInGradY(m_paddedPixels * inChan);
	std::vector<engineFloat> flatTarAct(m_paddedPixels * tarChan), flatTarGradX(m_paddedPixels * tarChan), flatTarGradY(m_paddedPixels * tarChan);
	// Raw normalized [0,1] pixel coordinates for the grid encoder, dataset.pixelX/Y are
	// assumed already in the coordMapper's input space if that space isn't [0,1],
	// normalize here (GridEncoding::FindCell clamps to [0,1], so anything outside it just
	// clamps to an edge cell silently)
	std::vector<engineFloat> flatPixelX(m_paddedPixels), flatPixelY(m_paddedPixels);

	for (int i = 0; i < (int)m_paddedPixels; ++i) {
		// Wrap around to the start of the image if we need extra pixels to fill the final batch
		size_t srcIdx = i % m_totalPixels;

		const auto& [inValues, inGradX, inGradY] = dataset.inputs[srcIdx];
		const auto& [tarValues, tarGradX, tarGradY] = dataset.targets[srcIdx];

		// We check if we actually have anything if not we just input 0 as our inputs layer 0 is not supposed to match the grid's 16 values
		size_t availableChans = inValues.size();
		size_t copyChans = (availableChans < inChan) ? availableChans : inChan;
		size_t inBytes = copyChans * sizeof(engineFloat);

		std::memcpy(&flatInAct[i * inChan], inValues.data(), inBytes);
		std::memcpy(&flatInGradX[i * inChan], inGradX.data(), inBytes);
		std::memcpy(&flatInGradY[i * inChan], inGradY.data(), inBytes);

		size_t tarBytes = tarChan * sizeof(engineFloat);
		std::memcpy(&flatTarAct[i * tarChan], tarValues.data(), tarBytes);
		std::memcpy(&flatTarGradX[i * tarChan], tarGradX.data(), tarBytes);
		std::memcpy(&flatTarGradY[i * tarChan], tarGradY.data(), tarBytes);

		flatPixelX[i] = dataset.pixelX[srcIdx];
		flatPixelY[i] = dataset.pixelY[srcIdx];
	}

	m_data->UploadData(flatInAct, flatInGradX, flatInGradY, flatTarAct, flatTarGradX, flatTarGradY, flatPixelX, flatPixelY);
}

void GpuTrainingPipeline::FlattenAndUploadDatasetChunked(size_t inChan, size_t tarChan, const SpatialDataset& dataset)
{
	// Reuseable Buffer (we use biggest img as upper bound)
    std::vector<engineFloat> buffer(m_paddedPixels * std::max(inChan, tarChan));

	// Helpers
    auto UploadInputChunk = [&](engineFloat* d_dest, auto extractVec) {
        for (int i = 0; i < (int)m_paddedPixels; ++i) {
			// Wrap around to the start of the image if we need extra pixels to fill the final batch
			size_t srcIdx = i % m_totalPixels;
            const auto& vec = extractVec(srcIdx);
			// We check if we actually have anything if not we just input 0 as our inputs layer 0 is not supposed to match the grid's 16 values
            const size_t copyChans = (vec.size() < inChan) ? vec.size() : inChan;
            
            // Zero the memory slot first, then copy the available floats
            std::memset(&buffer[i * inChan], 0, inChan * sizeof(engineFloat));
            std::memcpy(&buffer[i * inChan], vec.data(), copyChans * sizeof(engineFloat));
        }
        CUDA_CHECK(cudaMemcpy(d_dest, buffer.data(), m_paddedPixels * inChan * sizeof(engineFloat), cudaMemcpyHostToDevice));
    };

    auto UploadTargetChunk = [&](engineFloat* d_dest, size_t chanCount, auto extractData) {
	    const size_t bytes = chanCount * sizeof(engineFloat);
        for (int i = 0; i < (int)m_paddedPixels; ++i) {
        	// Wrap around to the start of the image if we need extra pixels to fill the final batch
			size_t srcIdx = i % m_totalPixels;
            std::memcpy(&buffer[i * chanCount], extractData(srcIdx), bytes);
        }
        CUDA_CHECK(cudaMemcpy(d_dest, buffer.data(), m_paddedPixels * bytes, cudaMemcpyHostToDevice));
    };

    auto UploadPixelChunk = [&](engineFloat* d_dest, auto extractVal) {
        for (int i = 0; i < (int)m_paddedPixels; ++i) {
        	// Wrap around to the start of the image if we need extra pixels to fill the final batch
			size_t srcIdx = i % m_totalPixels;
            buffer[i] = extractVal(srcIdx);
        }
        CUDA_CHECK(cudaMemcpy(d_dest, buffer.data(), m_paddedPixels * sizeof(engineFloat), cudaMemcpyHostToDevice));
    };

    // Upload Inputs
    UploadInputChunk(m_data->d_inputAct,   [&](size_t idx) -> const auto& { return dataset.inputs[idx].values; });
    UploadInputChunk(m_data->d_inputGradX, [&](size_t idx) -> const auto& { return dataset.inputs[idx].gradX; });
    UploadInputChunk(m_data->d_inputGradY, [&](size_t idx) -> const auto& { return dataset.inputs[idx].gradY; });

    // Upload Targets
    UploadTargetChunk(m_data->d_targetAct,   tarChan, [&](size_t idx) { return dataset.targets[idx].values.data(); });
    UploadTargetChunk(m_data->d_targetGradX, tarChan, [&](size_t idx) { return dataset.targets[idx].gradX.data(); });
    UploadTargetChunk(m_data->d_targetGradY, tarChan, [&](size_t idx) { return dataset.targets[idx].gradY.data(); });

    // Upload Pixels
    UploadPixelChunk(m_data->d_pixelX, [&](size_t idx) { return dataset.pixelX[idx]; });
    UploadPixelChunk(m_data->d_pixelY, [&](size_t idx) { return dataset.pixelY[idx]; });
}
