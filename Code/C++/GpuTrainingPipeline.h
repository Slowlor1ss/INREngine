#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "Types.h"
#include "ImageUtils.h"

class Network;
class GpuNetwork;
class GpuDataset;
struct SpatialDataset;

class GpuTrainingPipeline
{
public:
	// Uploads `dataset` (padded to a multiple of config::batch_size) and allocates the
	// render buffers used for live-preview/export frames at renderWidth x renderHeight
	GpuTrainingPipeline(Network& cpuNetwork, const SpatialDataset& dataset, size_t inChan, size_t tarChan,
	                    int renderWidth, int renderHeight,
	                    int sourceWidth, int sourceHeight,
	                    const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordMapper);
	~GpuTrainingPipeline();

	GpuTrainingPipeline(const GpuTrainingPipeline&) = delete;
	GpuTrainingPipeline& operator=(const GpuTrainingPipeline&) = delete;

	GpuNetwork& GetNetwork() { return *m_net; }
	
	engineFloat RunEpoch(size_t& currentImageIdx, size_t printEveryNBatches, size_t batchSize,
	                     engineFloat& learningRate, size_t& currentEpoch);

	// Runs inference across the whole render buffer and returns a flat RGB image
	std::vector<engineFloat> RenderFrame(int totalRenderPixels, size_t tarChan) const;

private:
    void FlattenAndUploadDataset(size_t inChan, size_t tarChan, const SpatialDataset& dataset);
	// v Called from FlattenAndUploadDataset v
	void FlattenAndUploadDatasetBluk(size_t inChan, size_t tarChan, const SpatialDataset& dataset);
    void FlattenAndUploadDatasetChunked(size_t inChan, size_t tarChan, const SpatialDataset& dataset);
	//

	std::unique_ptr<GpuNetwork> m_net;
	std::unique_ptr<GpuDataset> m_data;
	size_t m_inChan = 0;
	size_t m_tarChan = 0;
	size_t m_totalPixels = 0;
	size_t m_paddedPixels = 0;

	// Render buffers (owned)
	engineFloat* d_renderPixelX = nullptr;
	engineFloat* d_renderPixelY = nullptr;
	engineFloat* d_renderInputs = nullptr; // Fallback buffer for standard PE (non grid-encoding) rendering
	engineFloat* d_renderColors = nullptr;
	
	int m_sourceWidth = 0;
	int m_sourceHeight = 0;
};
