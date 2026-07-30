#include "ImageGenerator.h"
#include <thread>
#include <vector>
#include <algorithm>

#include "Network.h"

std::vector<float> GenerateReconstructedImage(
    const Network& network, 
    int width, 
    int height,
    const std::function<ImageUtils::SpatialData(float, float)>& coordinateMapper,
    RenderMode mode)
{
    std::vector<float> reconstructedImage(static_cast<size_t>(width) * height * 3);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    int rowsPerThread = (height + static_cast<int>(numThreads) - 1) / static_cast<int>(numThreads);

    for (unsigned int t = 0; t < numThreads; ++t)
    {
		int startY = static_cast<int>(t) * rowsPerThread;
		int endY = std::min(startY + rowsPerThread, height);

		if (startY >= height) break;

		// Note: We safely capture coordinateMapper by reference because we join all threads 
		// before this function returns, guaranteeing it won't go out of scope.
		workers.emplace_back([&network, width, height, startY, endY, &reconstructedImage, &coordinateMapper, mode]()
		{
			// TODO-Lkrikilion: clean this up its getting messy
			// Initalizing here to avoid recreating inside for loop, this still has a useless construction though based on the rendering mode
			// could probbaly be fixed with constexprs :D but honestly probably doesnt matter
		    std::vector<std::vector<float>> standardBuffers;
			Network::SpatialDerivativeBuffer spatialBuffers;


			for (int y = startY; y < endY; ++y)
			{
				float normY = static_cast<float>(y) / height;
				size_t rowOffset = static_cast<size_t>(y) * width * 3;

				for (int x = 0; x < width; ++x)
				{
					float normX = static_cast<float>(x) / width;

					// Reference whichever vector we ended up using
					//const std::vector<float>& finalInput = coordinateMapper ? mappedInput : fallbackInput;
					size_t pixelIndex = rowOffset + static_cast<size_t>(x) * 3;

					ImageUtils::SpatialData finalInput;
					if (coordinateMapper) 
					{
					    // PE is ON: Use the mapped values and their complex wave derivatives
						finalInput= coordinateMapper(normX, normY);
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
					    const std::vector<float>& output = network.PropagateThreadSafe(finalInput.values, standardBuffers);
					    reconstructedImage[pixelIndex + 0] = output[0];
					    reconstructedImage[pixelIndex + 1] = output[1];
					    reconstructedImage[pixelIndex + 2] = output[2];
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
					    reconstructedImage[pixelIndex + 0] = std::abs(finalGradX[0]) * 0.1f;	// Red = X Gradient
					    reconstructedImage[pixelIndex + 1] = std::abs(finalGradY[0]) * 0.1f;	// Green = Y Gradient
						reconstructedImage[pixelIndex + 2] = 0.0f;								// Blue = 0
					}
				}
			}
		});
    }

	for (auto& worker : workers)
	{
		if (worker.joinable())
		{
			worker.join();
		}
	}

    return reconstructedImage;
}
