#pragma once
#include <vector>
#include "Network.h"

// Parallelized RGB image reconstruction from network weights using multi-threading.
std::vector<float> GenerateReconstructedImage(const Network& network, int width, int height);
