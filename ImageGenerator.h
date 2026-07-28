#pragma once
#include <functional>
#include <vector>

class Network;

// Parallelized RGB image reconstruction from network weights using multi-threading.
std::vector<float> GenerateReconstructedImage(
    Network& network, 
    int width, 
    int height,
    const std::function<std::vector<float>(float, float)>& coordinateMapper = nullptr
);
