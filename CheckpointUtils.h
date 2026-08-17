#pragma once
#include <string>

#include "Network.h"
#include "GpuNetwork.h"

// Generates a checkpoint filename based on the input image filename.
// Example: "Image.bmp" -> "weights_biases_Sarah.csv"
std::string GetCheckpointFilename(const std::string& imageFilename, const std::string& outbasePath = "");

void LoadCheckpoint(Network& network, const std::string& filename);

void SaveCheckpoint(Network& network, GpuNetwork* gpuNet, const std::string& filename);
