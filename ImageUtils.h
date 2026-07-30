#pragma once
#include <vector>

class ImageUtils
{
public:
    struct SpatialData {
        std::vector<float> values; // Can be RGB colors, or Positional Encoding inputs
        std::vector<float> gradX;
        std::vector<float> gradY;
    };
    
    // Generates the Target Gradients for the new Cost Function using Central Finite Difference
    static std::vector<SpatialData> GenerateGradientTargets(
        const std::vector<std::vector<float>>& targetPixels, 
        int width, 
        int height);
};
