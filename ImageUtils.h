#pragma once
#include <vector>
#include <cmath>
#include <numbers>
#include <random>

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

class GaussianPositionalEncoder 
    {
    private:
        // The static random B matrix
        std::vector<std::pair<float, float>> m_B_matrix;
        float m_sigma;

    public:
        // Constructor generates the fixed random matrix ONCE
        GaussianPositionalEncoder(int numFrequencies, float sigma = 10.0f) : m_sigma(sigma) 
        {
            // CRITICAL: We use a fixed seed (e.g., 42) so the random matrix 
            // is exactly the same every time you run the program. 
            // Otherwise, your saved weights will break on the next launch!
            std::mt19937 gen(42); 
            std::normal_distribution<float> dist(0.0f, m_sigma);

            m_B_matrix.reserve(numFrequencies);
            for (int i = 0; i < numFrequencies; ++i) 
            {
                m_B_matrix.emplace_back(dist(gen), dist(gen));
            }
        }

        // The operator() allows this class to be passed directly into your std::function
        ImageUtils::SpatialData operator()(float x, float y) const 
        {
            ImageUtils::SpatialData data;
            data.values.reserve(m_B_matrix.size() * 2);
            data.gradX.reserve(m_B_matrix.size() * 2);
            data.gradY.reserve(m_B_matrix.size() * 2);

            constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;

            for (const auto& row : m_B_matrix) 
            {
                float bx = row.first;
                float by = row.second;
                
                // The dot product of the coordinate and the random frequency vector
                float theta = two_pi * (x * bx + y * by);

                float sin_val = std::sin(theta);
                float cos_val = std::cos(theta);

                // 1. The Values
                data.values.push_back(sin_val);
                data.values.push_back(cos_val);

                // 2. The X Derivatives (Chain Rule)
                data.gradX.push_back(cos_val * two_pi * bx);   // d/dx of sin
                data.gradX.push_back(-sin_val * two_pi * bx);  // d/dx of cos

                // 3. The Y Derivatives (Chain Rule)
                data.gradY.push_back(cos_val * two_pi * by);   // d/dy of sin
                data.gradY.push_back(-sin_val * two_pi * by);  // d/dy of cos
            }

            return data;
        }
    };
