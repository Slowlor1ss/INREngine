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

// Top-level orchestrator for a training run: loads the target image, builds the dataset
// and network, then runs the interactive training loop (CPU or GPU path, live viewer,
// checkpointing, and -- if config::benchmark_enabled -- metrics/frame logging) until the
// user quits or config::max_epochs is reached.
//
// This replaces the old free function `NeuralImageRecreator()`, which lived in a header
// that was #included into exactly one .cpp rather than compiled as its own translation
// unit -- that pattern doesn't scale past one .cpp and encouraged everything to pile up as
// loosely related global functions. This class + its sibling files (TrainingDataset,
// GpuTrainingPipeline, TrainingMetricsReporter, TrainingUserActions, CheckpointUtils) split
// that pile up along fairly natural seams: dataset building, GPU state ownership, reporting,
// user input, and checkpoint I/O each get their own small header/source pair, and this class
// wires them together.
class NeuralImageRecreator
{
public:
	NeuralImageRecreator();

	void Run();

private:
	// Everything that needs to exist before the Network can be constructed (its layer dims
	// depend on the loaded image + coordinate mapper), bundled so it can all be built in one
	// shot in the member-initializer list, ahead of m_network.
	struct InitData
	{
		BMPParsedData data;
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
	std::unique_ptr<GpuTrainingPipeline> m_gpu;
	std::unique_ptr<ImageWindow> m_window;

	size_t m_inChan = 0;
	size_t m_tarChan = 0;
	int m_renderWidth = 0;
	int m_renderHeight = 0;

	size_t m_currentImage = 0;
	size_t m_currentEpoch = 0;
	engineFloat m_learningRate = 0.0f;
	bool m_liveUpdateWindow = true;
};