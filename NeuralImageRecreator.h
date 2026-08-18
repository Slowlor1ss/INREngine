#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Types.h"
#include "Network.h"
#include "TrainingThreadPool.h"
#include "WindowRenderer.h"
#include "Gemini/BMPParser.h"
#include "ImageUtils.h"
#include "TrainingDataset.h"
#include "GpuTrainingPipeline.h"
#include "PythonVisualizerBridge.h"

class NeuralImageRecreator
{
public:
	NeuralImageRecreator();

	void Run();

private:
	struct InitData
	{
		ImgParser::ImageParsedData data;
		std::string weightsFile;
		std::vector<engineFloat> flatTargetImage;
		std::vector<engineFloat> flatHDImage;
		std::function<ImageUtils::SpatialData(engineFloat, engineFloat)> coordMapper;
		SpatialDataset dataset;
		size_t inputLayerSize = 0;
	};
	static InitData LoadInitData();
	static std::vector<size_t> BuildLayerDims(size_t inputLayerSize);

	engineFloat RunCpuEpoch(size_t& currentImageIdx, size_t printEveryNBatches, size_t batchSize, engineFloat& learningRate);
	std::vector<engineFloat> RenderLiveFrame(int totalRenderPixels, size_t tarChan);
	void SaveFinalOutputs(GpuNetwork* gpuNet);

private:
	InitData m_init;
	Network m_network;
	TrainingThreadPool m_threadPool;
	std::unique_ptr<GpuTrainingPipeline> m_gpu = nullptr;
	std::unique_ptr<ImageWindow> m_window = nullptr;
	std::unique_ptr<PythonVisualizerBridge> m_pyViz = nullptr; // optional, null-safe

	size_t m_inChan = 0;
	size_t m_tarChan = 0;
	int m_renderWidth = 0;
	int m_renderHeight = 0;

	size_t m_currentImage = 0;
	size_t m_currentEpoch = 0;
	engineFloat m_learningRate = 0.0f;
	bool m_liveUpdateWindow = true;
};