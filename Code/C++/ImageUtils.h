#pragma once
#include <vector>
#include <cmath>
#include <numbers>
#include <utility>

#include "Types.h"

class ImageUtils
{
public:
    struct SpatialData {
        std::vector<engineFloat> values; // Can be RGB colors, or Positional Encoding inputs
        std::vector<engineFloat> gradX;
        std::vector<engineFloat> gradY;
    };
    
    // Generates the Target Gradients for the new Cost Function using Central Finite Difference
    static std::vector<SpatialData> GenerateGradientTargets(
        const std::vector<std::vector<engineFloat>>& targetPixels, 
        int width, 
        int height);

    struct ImageMetrics {
		engineFloat mse;
	    engineFloat rmse;
	    engineFloat mae;
	    engineFloat ssim;
	    engineFloat psnr;
	};

    static ImageMetrics CalculateFullImageMetrics(
        const std::vector<engineFloat>& predicted,
        const std::vector<engineFloat>& target,
        int channels = 3); // Defaulting to RGB);

    // Expands a coordinate (x, y) into multiple frequency bands with decay, along with the
    // analytic derivatives of each band w.r.t. x and y
    static SpatialData PositionalEncodeWithDerivatives(engineFloat x, engineFloat y, int numFrequencies);

    // Fixed random Fourier feature encoder ("Gaussian" positional encoding)
    class GaussianPositionalEncoder
    {
    public:
        // Generates the fixed random matrix
        GaussianPositionalEncoder(int numFrequencies, engineFloat sigma = 10.0);

        // operator() allows this class to be passed directly into your std::function
        SpatialData operator()(engineFloat x, engineFloat y) const;

    private:
        // static random B matrix
        std::vector<std::pair<engineFloat, engineFloat>> m_B_matrix;
        engineFloat m_sigma;
    };
};