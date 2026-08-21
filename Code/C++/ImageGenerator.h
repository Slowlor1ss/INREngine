#pragma once
#include <functional>
#include <vector>

#include "ImageUtils.h"
#include "TrainingThreadPool.h"

class Network;

enum class RenderMode {
    // Standard view
	StandardRGB,        // Normal forward pass

    // SpatialGradient is usefull for debugging:
	// Red: X-Gradient
    // = very fast horizontal change (so most common on vertical edges)
    // Green: Y-Gradient
    // = fast vertically change (so most common on horizontal edges)
    // Yellow (Red + Green): happens when Red and Green channels are both showing (doh)
    // = fast change in both directions
    SpatialGradient,    // First derivative

    // TODO-Lkrikilion:
    //Laplacian,          // Second derivative
    Blur,
    Last
};

// struct MappedInput {
//     std::vector<engineFloat> values;
//     std::vector<engineFloat> gradX;
//     std::vector<engineFloat> gradY;
// };

// Parallelized RGB image reconstruction from network weights using multi-threading.
// std::vector<engineFloat> GenerateReconstructedImage(
//     const Network& network, 
//     int width, 
//     int height,
//     const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordinateMapper = nullptr,
//     RenderMode mode = RenderMode::StandardRGB
// );

std::vector<engineFloat> GenerateReconstructedImage(
    const Network& network, 
    int width, 
    int height,
    const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordinateMapper,
    RenderMode mode,
    TrainingThreadPool& threadPool);
