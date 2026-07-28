#include "ImageGenerator.h"
#include <thread>
#include <vector>
#include <algorithm>

#include "Network.h"

std::vector<float> GenerateReconstructedImage(
    const Network& network, 
    int width, 
    int height,
    const std::function<std::vector<float>(float, float)>& coordinateMapper)
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
       workers.emplace_back([&network, width, height, startY, endY, &reconstructedImage, &coordinateMapper]()
       {
          std::vector<std::vector<float>> threadBuffers;
          
          // Pre-allocate the fallback vector to prevent reallocations inside the loop
          std::vector<float> fallbackInput(2);
          std::vector<float> mappedInput;

          for (int y = startY; y < endY; ++y)
          {
             float normY = static_cast<float>(y) / height;
             size_t rowOffset = static_cast<size_t>(y) * width * 3;

             for (int x = 0; x < width; ++x)
             {
                float normX = static_cast<float>(x) / width;

                // Apply positional encoding if provided, otherwise use standard [x, y]
                if (coordinateMapper)
                {
                    mappedInput = coordinateMapper(normX, normY);
                }
                else
                {
                    fallbackInput[0] = normX;
                    fallbackInput[1] = normY;
                }

                // Reference whichever vector we ended up using
                const std::vector<float>& finalInput = coordinateMapper ? mappedInput : fallbackInput;

                const std::vector<float>& output = network.PropagateThreadSafe(finalInput, threadBuffers);

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
