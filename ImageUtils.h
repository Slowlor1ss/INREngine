#pragma once
#include <vector>
#include <cmath>
#include <numbers>
#include <random>

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
};

// TODO: Move this to its own util file together with our normal Positional Encoding
class GaussianPositionalEncoder 
    {
    private:
        // static random B matrix
        std::vector<std::pair<engineFloat, engineFloat>> m_B_matrix;
        engineFloat m_sigma;

    public:
        // Generates the fixed random matrix
        GaussianPositionalEncoder(int numFrequencies, engineFloat sigma = 10.0) : m_sigma(sigma) 
        {
            // We use a fixed seed (e.g., 42) so the random matrix 
            // is exactly the same every time we run the program. 
            // Otherwise, our saved weights will break on the next launch
            std::mt19937 gen(42); // DO NOT CHANGE
            std::normal_distribution<engineFloat> dist(0.0f, m_sigma);

            m_B_matrix.reserve(numFrequencies);
            for (int i = 0; i < numFrequencies; ++i) 
            {
                m_B_matrix.emplace_back(dist(gen), dist(gen));
            }
        }

        // operator() allows this class to be passed directly into your std::function
        ImageUtils::SpatialData operator()(engineFloat x, engineFloat y) const 
        {
            ImageUtils::SpatialData data;
            data.values.reserve(m_B_matrix.size() * 2);
            data.gradX.reserve(m_B_matrix.size() * 2);
            data.gradY.reserve(m_B_matrix.size() * 2);

            constexpr engineFloat two_pi = 2.0 * std::numbers::pi_v<engineFloat>;

            for (const auto& row : m_B_matrix) 
            {
                engineFloat bx = row.first;
                engineFloat by = row.second;
                
                // Dot product of the coordinate and the random frequency vector
                engineFloat theta = two_pi * (x * bx + y * by);

                engineFloat sin_val = std::sin(theta);
                engineFloat cos_val = std::cos(theta);

                // Values
                data.values.push_back(sin_val);
                data.values.push_back(cos_val);

                // X Derivatives (Chain Rule)
                data.gradX.push_back(cos_val * two_pi * bx);   // d/dx of sin
                data.gradX.push_back(-sin_val * two_pi * bx);  // d/dx of cos

                // Y Derivatives (Chain Rule)
                data.gradY.push_back(cos_val * two_pi * by);   // d/dy of sin
                data.gradY.push_back(-sin_val * two_pi * by);  // d/dy of cos
            }

            return data;
        }
    };
