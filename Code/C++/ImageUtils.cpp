#include "ImageUtils.h"
#include <random>

std::vector<ImageUtils::SpatialData> ImageUtils::GenerateGradientTargets(
    const std::vector<std::vector<engineFloat>>& targetPixels, int width, int height)
{
    std::vector<SpatialData> dataset(width * height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            dataset[idx].values = targetPixels[idx];
			
            // Initialize gradient vectors for R, G, B channels
            dataset[idx].gradX.resize(3, 0.0);
            dataset[idx].gradY.resize(3, 0.0);

            // Safely grab neighbor indices (clamping at the borders to prevent out-of-bounds)
            int x_prev = std::max(0, x - 1);
            int x_next = std::min(width - 1, x + 1);
            int y_prev = std::max(0, y - 1);
            int y_next = std::min(height - 1, y + 1);

            int idx_x_prev = y * width + x_prev;
            int idx_x_next = y * width + x_next;
            int idx_y_prev = y_prev * width + x;
            int idx_y_next = y_next * width + x;

            // Calculate the spatial derivative for each color channel (R, G, B)
            for (int c = 0; c < 3; ++c)
            {
                // Multiply by (width / 2.0f) and (height / 2.0f) to map the pixel-space gradient to normalized [-1, 1] coordinate space!
                dataset[idx].gradX[c] = ((targetPixels[idx_x_next][c] - targetPixels[idx_x_prev][c]) / (engineFloat)2.0) * (width / (engineFloat)2.0);
				dataset[idx].gradY[c] = ((targetPixels[idx_y_next][c] - targetPixels[idx_y_prev][c]) / (engineFloat)2.0) * (height / (engineFloat)2.0);
            }
        }
    }
    return dataset;
}

ImageUtils::ImageMetrics ImageUtils::CalculateFullImageMetrics(const std::vector<engineFloat>& predicted,
	const std::vector<engineFloat>& target, int channels)
{
	ImageMetrics metrics = {0, 0, 0, 0, 0};
    if (predicted.size() != target.size() || predicted.empty()) return metrics;

    size_t totalElements = predicted.size();
    size_t numPixels = totalElements / channels;
    
    engineFloat sumErrorSq = 0.0f;
    engineFloat sumAbsError = 0.0f;

    // Calculate Global Error Metrics (MSE, RMSE, MAE, PSNR)
    for (size_t i = 0; i < totalElements; ++i) {
        engineFloat err = predicted[i] - target[i];
        sumErrorSq += (err * err);
        sumAbsError += std::abs(err);
    }
    
    metrics.mse = sumErrorSq / static_cast<engineFloat>(totalElements);
    metrics.rmse = std::sqrt(metrics.mse);
    metrics.mae = sumAbsError / static_cast<engineFloat>(totalElements);
    metrics.psnr = (metrics.mse <= 0.0000001f) ? 100.0f : -10.0f * std::log10(metrics.mse);

    // Calculate Per-Channel SSIM
    engineFloat totalSSIM = 0.0f;
    constexpr engineFloat c1 = 0.0001f; // (0.01 * 1.0)^2
    constexpr engineFloat c2 = 0.0009f; // (0.03 * 1.0)^2

    for (int c = 0; c < channels; ++c) {
        engineFloat meanPred = 0.0f;
        engineFloat meanTar = 0.0f;

        // Calculate means for this specific channel
        for (size_t p = 0; p < numPixels; ++p) {
            size_t idx = p * channels + c;
            meanPred += predicted[idx];
            meanTar += target[idx];
        }
        meanPred /= static_cast<engineFloat>(numPixels);
        meanTar /= static_cast<engineFloat>(numPixels);

        // Calculate variance and covariance for this specific channel
        engineFloat varPred = 0.0f;
        engineFloat varTar = 0.0f;
        engineFloat covar = 0.0f;

        for (size_t p = 0; p < numPixels; ++p) {
            size_t idx = p * channels + c;
            engineFloat pDiff = predicted[idx] - meanPred;
            engineFloat tDiff = target[idx] - meanTar;
            
            varPred += (pDiff * pDiff);
            varTar += (tDiff * tDiff);
            covar += (pDiff * tDiff);
        }
        varPred /= static_cast<engineFloat>(numPixels);
        varTar /= static_cast<engineFloat>(numPixels);
        covar /= static_cast<engineFloat>(numPixels);

        // SSIM Formula for this channel
        engineFloat numerator = (2.0f * meanPred * meanTar + c1) * (2.0f * covar + c2);
        engineFloat denominator = (meanPred * meanPred + meanTar * meanTar + c1) * (varPred + varTar + c2);
        
        totalSSIM += (numerator / denominator);
    }

    // Average the SSIM across the channels
    metrics.ssim = totalSSIM / static_cast<engineFloat>(channels);

    return metrics;
}

// TODO: merge with GaussianPositionalEncoder::operator() or something, this is bad but
// we need one for the image dataset and another for the coord mapper
ImageUtils::SpatialData ImageUtils::PositionalEncodeWithDerivatives(engineFloat x, engineFloat y, int numFrequencies)
{
    SpatialData result;
    const size_t size = static_cast<size_t>(numFrequencies) * 4;
    result.values.reserve(size);
    result.gradX.reserve(size);
    result.gradY.reserve(size);

    constexpr engineFloat k_pi = std::numbers::pi_v<engineFloat>;

    for (int i = 0; i < numFrequencies; ++i) {
        const engineFloat freq = std::pow((engineFloat)2.0, (engineFloat)i) * k_pi;
        const engineFloat weight = 1.0f - (static_cast<engineFloat>(i) / static_cast<engineFloat>(numFrequencies));

        // Pre-calculate to save CPU cycles
        engineFloat sin_x = std::sin(x * freq);
        engineFloat cos_x = std::cos(x * freq);
        engineFloat sin_y = std::sin(y * freq);
        engineFloat cos_y = std::cos(y * freq);

        // Standard Values
        result.values.push_back(sin_x * weight);
        result.values.push_back(cos_x * weight);
        result.values.push_back(sin_y * weight);
        result.values.push_back(cos_y * weight);

        // X Gradients (d/dx)
        result.gradX.push_back(freq * cos_x * weight);  // d/dx sin = cos * freq
        result.gradX.push_back(-freq * sin_x * weight); // d/dx cos = -sin * freq
        result.gradX.push_back(0.0f);                   // d/dx of Y is 0
        result.gradX.push_back(0.0f);

        // Y Gradients (d/dy)
        result.gradY.push_back(0.0f);                   // d/dy of X is 0
        result.gradY.push_back(0.0f);
        result.gradY.push_back(freq * cos_y * weight);
        result.gradY.push_back(-freq * sin_y * weight);
    }
    return result;
}

ImageUtils::GaussianPositionalEncoder::GaussianPositionalEncoder(int numFrequencies, engineFloat sigma) : m_sigma(sigma)
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

ImageUtils::SpatialData ImageUtils::GaussianPositionalEncoder::operator()(engineFloat x, engineFloat y) const
{
    SpatialData data;
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