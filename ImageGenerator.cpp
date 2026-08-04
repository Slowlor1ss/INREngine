#include "ImageGenerator.h"
#include <vector>
#include <algorithm>

#include "Network.h"

std::vector<engineFloat> GenerateReconstructedImage(
    const Network& network, 
    int width, 
    int height,
    const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordinateMapper,
    RenderMode mode,
    TrainingThreadPool& threadPool)
{
    size_t totalPixels = static_cast<size_t>(width) * height;
    std::vector<engineFloat> reconstructedImage(totalPixels * 3);

    // Pre-allocate buffers for each thread so they don't recreate them inside the loop
    // or overwrite each other's activations during parallel execution
    size_t numWorkers = threadPool.GetThreadCount();
    std::vector<std::vector<std::vector<engineFloat>>> threadStandardBuffers(numWorkers);
    std::vector<Network::SpatialDerivativeBuffer> threadSpatialBuffers(numWorkers);

    // Execute via the persistent thread pool
    threadPool.RunParallelTask(totalPixels, [&](size_t threadIndex, size_t startIdx, size_t endIdx)
    {
        // Grab this specific thread's pre-allocated buffers
        auto& standardBuffers = threadStandardBuffers[threadIndex];
        auto& spatialBuffers = threadSpatialBuffers[threadIndex];

        // Loop over the flat 1D chunk assigned to this thread
        for (size_t i = startIdx; i < endIdx; ++i)
        {
            // Convert flat 1D index back to 2D x,y coordinates
            int x = static_cast<int>(i % width);
            int y = static_cast<int>(i / width);

            // Changes 0.0 -> 1.0 into -1.0 -> 1.0
            engineFloat normY = (static_cast<engineFloat>(y) / (engineFloat)height) * 2.0f - 1.0f;
            engineFloat normX = (static_cast<engineFloat>(x) / (engineFloat)width) * 2.0f - 1.0f;
            
            size_t pixelIndex = i * 3;

            ImageUtils::SpatialData finalInput;
            if (coordinateMapper) 
            {
                // PE is ON: Use the mapped values and their complex wave derivatives
                finalInput = coordinateMapper(normX, normY);
            }
            else 
            {
                // PE is OFF: Standard Inputs and base derivatives
                finalInput.values = { normX, normY };
                finalInput.gradX  = { 1.0f, 0.0f }; // d/dx of X is 1, d/dx of Y is 0
                finalInput.gradY  = { 0.0f, 1.0f }; // d/dy of X is 0, d/dy of Y is 1
            }

            if (mode == RenderMode::StandardRGB)
            {
                const std::vector<engineFloat>& output = network.PropagateThreadSafe(finalInput.values, standardBuffers);

                reconstructedImage[pixelIndex + 0] = output[0];
                reconstructedImage[pixelIndex + 1] = output[1];
                reconstructedImage[pixelIndex + 2] = output[2];
            }
            else if (mode == RenderMode::Blur)
            {
                // 1. Query the exact center of the pixel
                const std::vector<engineFloat>& centerOutput = network.PropagateThreadSafe(finalInput.values, standardBuffers);
                engineFloat centerR = centerOutput[0];
                engineFloat centerG = centerOutput[1];
                engineFloat centerB = centerOutput[2];

                // 2. Setup the Continuous Bilateral Filter parameters
                // We sample sub-pixel distances (e.g., 1/3rd of a pixel away)
                const engineFloat subPixelDistMul = 2.5f;
                engineFloat subPixelDistX = (1.0f / width) * subPixelDistMul;
                engineFloat subPixelDistY = (1.0f / height) * subPixelDistMul;
                
                // Sigma values control how aggressive the filter is. 
                // Lower colorSigma preserves edges better.
                engineFloat colorSigma = 0.1f; 

                engineFloat sumR = centerR;
                engineFloat sumG = centerG;
                engineFloat sumB = centerB;
                engineFloat sumWeight = 1.0f;

                // 3. MORE SAMPLES: Check 8 directions instead of 4 (including diagonals)
                std::vector<std::pair<engineFloat, engineFloat>> subPixelOffsets = {
                    { subPixelDistX, 0.0f }, { -subPixelDistX, 0.0f },
                    { 0.0f, subPixelDistY }, { 0.0f, -subPixelDistY },
                    { subPixelDistX, subPixelDistY }, { -subPixelDistX, -subPixelDistY },
                    { subPixelDistX, -subPixelDistY }, { -subPixelDistX, subPixelDistY }
                };

                for (const auto& offset : subPixelOffsets)
                {
                    // Generate the sub-pixel input
                    ImageUtils::SpatialData subInput;
                    if (coordinateMapper) {
                        subInput = coordinateMapper(normX + offset.first, normY + offset.second);
                    } else {
                        subInput.values = { normX + offset.first, normY + offset.second };
                    }

                    // Query the network for this continuous sub-coordinate
                    const std::vector<engineFloat>& subOut = network.PropagateThreadSafe(subInput.values, standardBuffers);

                    // Calculate color distance (Euclidean distance between RGB values)
                    engineFloat colorDistSq = (subOut[0] - centerR) * (subOut[0] - centerR) +
                                        (subOut[1] - centerG) * (subOut[1] - centerG) +
                                        (subOut[2] - centerB) * (subOut[2] - centerB);

                    // Calculate Bilateral Weight (spatial weight is constant here since distances are equal, 
                    // so we only penalize based on color difference)
                    engineFloat weight = std::exp(-colorDistSq / (2.0f * colorSigma * colorSigma));

                    // Accumulate
                    sumR += subOut[0] * weight;
                    sumG += subOut[1] * weight;
                    sumB += subOut[2] * weight;
                    sumWeight += weight;
                }

                // 4. Write the final denoised, edge-preserved pixel
                reconstructedImage[pixelIndex + 0] = sumR / sumWeight;
                reconstructedImage[pixelIndex + 1] = sumG / sumWeight;
                reconstructedImage[pixelIndex + 2] = sumB / sumWeight;
            }
            else if (mode == RenderMode::SpatialGradient)
            {
                network.PropagateSpatialDerivativesThreadSafe(
                    finalInput.values, 
                    finalInput.gradX, 
                    finalInput.gradY, 
                    spatialBuffers);
                
                // Grab the X and Y gradients of the Red channel (index 0) from the final layer
                const auto& finalGradX = spatialBuffers.gradientX.back();
                const auto& finalGradY = spatialBuffers.gradientY.back();
                
                // We take the absolute value so negative gradients dont render as black
                // We scale it by a small factor (like 0.1) so the colors arent blown out to pure white
                //reconstructedImage[pixelIndex + 0] = std::abs(finalGradX[0]) * 0.1f; // Red = X Gradient
                //reconstructedImage[pixelIndex + 1] = std::abs(finalGradY[0]) * 0.1f; // Green = Y Gradient

                // We divide by (width / 2.0f) to map the normalized gradient back to pixel space for viewing
                reconstructedImage[pixelIndex + 0] = std::abs(finalGradX[0] / (width / 2.0f));  
                reconstructedImage[pixelIndex + 1] = std::abs(finalGradY[0] / (height / 2.0f));

                reconstructedImage[pixelIndex + 2] = 0.0f;                      // Blue = 0
            }
        }
    });

    return reconstructedImage;
}