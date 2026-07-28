#pragma once
#include <vector>
#include "Network.h"

// Parallelized RGB image reconstruction from network weights using multi-threading.
std::vector<float> GenerateReconstructedImageMNIST(Network& network, const std::vector<float>& inputs);
std::vector<float> GenerateImageFromLatent(Network& network, float x, float y, int scale = 10);
std::vector<float> GenerateReconstructedImage(const Network& network, int width, int height);
