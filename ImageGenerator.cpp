#include "ImageGenerator.h"
#include <thread>
#include <vector>
#include <algorithm>

std::vector<float> GenerateReconstructedImage(const Network& network, int width, int height)
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

		workers.emplace_back([&network, width, height, startY, endY, &reconstructedImage]()
		{
			std::vector<std::vector<float>> threadBuffers;
			std::vector<float> input(2);

			for (int y = startY; y < endY; ++y)
			{
				float normY = static_cast<float>(y) / height;
				input[1] = normY;
				size_t rowOffset = static_cast<size_t>(y) * width * 3;

				for (int x = 0; x < width; ++x)
				{
					input[0] = static_cast<float>(x) / width;

					const std::vector<float>& output = network.PropagateThreadSafe(input, threadBuffers);

					size_t pixelIndex = rowOffset + static_cast<size_t>(x) * 3;
					reconstructedImage[pixelIndex + 0] = output[0]; // R
					reconstructedImage[pixelIndex + 1] = output[1]; // G
					reconstructedImage[pixelIndex + 2] = output[2]; // B
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
