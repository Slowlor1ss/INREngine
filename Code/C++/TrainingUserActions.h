#pragma once
#include <cstdint>
#include <functional>
#include <string>

#include "Network.h"
#include "GpuNetwork.h"
#include "Gemini/BMPParser.h"
#include "TrainingThreadPool.h"
#include "ImageUtils.h"

enum class UserAction : uint8_t {
	None,
	SaveWeights,
	ExportImage,
	ToggleViewer,
	SwapRenderMode,
	Quit,
};

// Prints the interactive keybinding help shown once training starts.
void PrintControls();

// Polls non-blocking keyboard input buffer and maps it to a UserAction.
UserAction PollUserAction();

// Handles user actions outside the main training loop; returns false to break the loop.
bool HandleUserAction(UserAction action, Network& network, GpuNetwork* gpuNet, const ImgParser::ImageParsedData& data,
                       const std::string& weightsFile, bool& liveUpdateWindow,
                       const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& mapper,
                       TrainingThreadPool& threadPool);
