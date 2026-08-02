#include "ImageUtils.h"

std::vector<ImageUtils::SpatialData> ImageUtils::GenerateGradientTargets(
    const std::vector<std::vector<float>>& targetPixels, int width, int height)
{
    std::vector<SpatialData> dataset(width * height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            dataset[idx].values = targetPixels[idx];
			
            // Initialize gradient vectors for R, G, B channels
            dataset[idx].gradX.resize(3, 0.0f);
            dataset[idx].gradY.resize(3, 0.0f);

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
                dataset[idx].gradX[c] = ((targetPixels[idx_x_next][c] - targetPixels[idx_x_prev][c]) / 2.0f) * (width / 2.0f);
				dataset[idx].gradY[c] = ((targetPixels[idx_y_next][c] - targetPixels[idx_y_prev][c]) / 2.0f) * (height / 2.0f);
            }
        }
    }
    return dataset;
}
