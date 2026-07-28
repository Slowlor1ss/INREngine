#include "ImageGenerator.h"
#include <thread>
#include <vector>
#include <algorithm>

#include "Network.h"

std::vector<float> GenerateReconstructedImageMNIST(Network& network, const std::vector<float>& originalInputs)
{
    const int originalWidth = 28;
    const int height = 28;

    // The new canvas is twice as wide: 56 * 28 pixels
    const int combinedWidth = originalWidth * 2;

    // Total size: 56 width * 28 height * 3 color channels = 4704 elements
    std::vector<float> doubleImage(static_cast<size_t>(combinedWidth) * height * 3);

    // 1. Get the 784 reconstructed pixels from the network
    const std::vector<float>& mnistOutputs = network.Propagate(originalInputs);

    // 2. Loop through every row and column of the new, wider grid
    for (int y = 0; y < height; ++y)
    {
        // Calculate where the current row starts in the big combined buffer
        size_t combinedRowOffset = static_cast<size_t>(y) * combinedWidth * 3;

        // Calculate where the current row starts in the individual 28x28 source images
        size_t sourceRowOffset = static_cast<size_t>(y) * originalWidth;

        for (int x = 0; x < combinedWidth; ++x)
        {
            float grayscaleValue = 0.0f;

            // LEFT SIDE: Columns 0 to 27 (Original Image)
            if (x < originalWidth)
            {
                size_t pixelIdx = sourceRowOffset + x;
                if (pixelIdx < originalInputs.size()) {
                    grayscaleValue = originalInputs[pixelIdx];
                }
            }
            // RIGHT SIDE: Columns 28 to 55 (Reconstructed Image)
            else
            {
                int localX = x - originalWidth; // Map 28->0, 29->1, etc.
                size_t pixelIdx = sourceRowOffset + localX;
                if (pixelIdx < mnistOutputs.size()) {
                    grayscaleValue = mnistOutputs[pixelIdx];
                }
            }

            // Write the RGB values into the woven row buffer
            size_t combinedPixelIndex = combinedRowOffset + static_cast<size_t>(x) * 3;
            doubleImage[combinedPixelIndex + 0] = grayscaleValue; // R
            doubleImage[combinedPixelIndex + 1] = grayscaleValue; // G
            doubleImage[combinedPixelIndex + 2] = grayscaleValue; // B
        }
    }

    return doubleImage;
}

#pragma optimize("", off)
std::vector<float> GenerateImageFromLatent(Network& network, float x, float y, int scale)
{
    const int originalSize = 28;
    const int scaledSize = originalSize * scale;

    // Scaled RGB image buffer
    std::vector<float> rgbImage(static_cast<size_t>(scaledSize) * scaledSize * 3);

    // 1. Push coordinates into the amputated decoder
    std::vector<float> latentInput = { x, y };
    const std::vector<float>& output = network.Propagate(latentInput);

    // 2. Map 28x28 grayscale to the scaled RGB window buffer
    for (int sy = 0; sy < scaledSize; ++sy)
    {
        for (int sx = 0; sx < scaledSize; ++sx)
        {
            // Nearest neighbor scaling (keeps the pixels crisp)
            int origX = sx / scale;
            int origY = sy / scale;
            int origIdx = origY * originalSize + origX;

            float val = (origIdx < output.size()) ? output[origIdx] : 0.0f;

            size_t px = (sy * scaledSize + sx) * 3;
            rgbImage[px + 0] = val; // R
            rgbImage[px + 1] = val; // G
            rgbImage[px + 2] = val; // B
        }
    }
    return rgbImage;
}

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
