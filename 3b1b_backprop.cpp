#include "Network.h"
#include "WindowRenderer.h"
#include "ImageGenerator.h"
#include "CostFuncDataBase.h"
#include "Gemini/BMPParser.h"

// Alternative dataset readers (available for switching modes)
#include "Gemini/MNISTReader.h"
#include "Gemini/CustomFileReader.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <conio.h> // For _kbhit() and _getch()

// TODO: don't see the point for the database, just use them directly? to have a single instance? does that matter?
// seems to be to use the inheritance more easily >> convert to template instead?
// 
// TODO: Relu needs HE initialization: make it so that happens automatically when selecting relu, and not when selecting sigmoid for a layer.
// TODO: serialization
// TODO: better interface / frontend.

// ============================================================================
// Alternative Dataset & Mode Examples (Uncomment to use)
// ============================================================================
/*
// ----------------------------------------------------------------------------
// 1. MNIST Dataset Loading Example
// ----------------------------------------------------------------------------
// To train on MNIST digit recognition instead of image regression:
// Network layerDims for MNIST: { 784, 64, 64, 10 }
//
static void LoadMNISTData(std::vector<std::vector<float>>& trainImages,
                          std::vector<std::vector<float>>& trainLabels,
                          std::vector<std::vector<float>>& testImages,
                          std::vector<std::vector<float>>& testLabels)
{
    trainImages = read_mnist_images("DATA/train-images-idx3-ubyte");
    trainLabels = read_mnist_labels("DATA/train-labels-idx1-ubyte");
    testImages  = read_mnist_images("DATA/t10k-images-idx3-ubyte");
    testLabels  = read_mnist_labels("DATA/t10k-labels-idx1-ubyte");
}

// ----------------------------------------------------------------------------
// 2. Custom Binary File Reader Example
// ----------------------------------------------------------------------------
static void LoadCustomBinaryData(const char* filename, ParsedData& outData)
{
    ParseBinaryData(filename, outData);
}

// ----------------------------------------------------------------------------
// 3. Alternative Image Targets & Iteration Methods
// ----------------------------------------------------------------------------
// ParseBMPData("light_on.bmp", data);
// Sequential training image traversal:
// currentImage = (currentImage + 1) % images.size();
*/

// ============================================================================
// Input & State Definitions
// ============================================================================

enum class UserAction : uint8_t {
	None,
	SaveWeights,
	ExportImage,
	ToggleViewer,
	Quit
};

namespace
{
// ============================================================================
// Helper Functions
// ============================================================================

// Generates a checkpoint filename based on the input image filename.
// Example: "Sarah.bmp" -> "weights_biases_Sarah.csv"
static std::string GetCheckpointFilename(const std::string& imagePath)
{
	size_t lastSlash = imagePath.find_last_of("/\\");
	std::string filename = (lastSlash == std::string::npos) ? imagePath : imagePath.substr(lastSlash + 1);

	size_t lastDot = filename.find_last_of('.');
	std::string stem = (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);

	return "weights_biases_" + stem + ".csv";
}

static void LoadCheckpoint(Network& network, const std::string& filename)
{
	std::ifstream inFile{ filename };
	if (inFile.is_open())
	{
		network.Deserialize(inFile);
	}
	else
	{
		std::cout << "No existing checkpoint found (" << filename << "). Starting with fresh weights.\n";
	}
}

static void SaveCheckpoint(Network& network, const std::string& filename)
{
	std::ofstream outFile{ filename };
	if (outFile.is_open())
	{
		network.Serialize(outFile);
		std::cout << "Weights saved to " << filename << "!\n";
	}
}

static void PrintControls()
{
	std::cout << "Training started.\n"
		<< " [Q/ESC] - Stop training\n"
		<< " [W]     - Save weights\n"
		<< " [E]     - Save image to disk\n"
		<< " [V]     - Toggle live viewer window\n\n";
}

// Polls non-blocking keyboard input buffer and maps to UserAction.
static UserAction PollUserAction()
{
	if (!_kbhit()) return UserAction::None;

	int ch = _getch();
	switch (ch)
	{
		case 'q': case 'Q': case 27: // ESC
			return UserAction::Quit;
		case 'w': case 'W':
			return UserAction::SaveWeights;
		case 'e': case 'E':
			return UserAction::ExportImage;
		case 'v': case 'V':
			return UserAction::ToggleViewer;
		default:
			return UserAction::None;
	}
}

// Handles user actions outside the main training loop; returns false to break loop.
static bool HandleUserAction(const UserAction action, Network& network, const BMPParsedData& data, const std::string& weightsFile, bool& liveUpdateWindow)
{
	switch (action)
	{
		case UserAction::Quit:
			std::cout << "\n[Interrupt Received] Stopping training...\n";
			return false;

		case UserAction::SaveWeights:
			SaveCheckpoint(network, weightsFile);
			break;

		case UserAction::ExportImage:
		{
			std::vector<float> reconstructedImage = GenerateReconstructedImage(network, data.width, data.height);
			saveBMP("network_output.bmp", data.width, data.height, reconstructedImage);
			std::cout << "Successfully saved network_output.bmp!\n";
			break;
		}

		case UserAction::ToggleViewer:
			liveUpdateWindow = !liveUpdateWindow;
			std::cout << (liveUpdateWindow ? "Live viewer window ENABLED\n" : "Live viewer window DISABLED\n");
			break;

		case UserAction::None:
			break;
	}

	return true;
}

// Generates a 32-bit random index to correctly sample datasets larger than RAND_MAX (32,767).
static size_t GetRandomImageIndex(const size_t totalImages)
{
	static thread_local std::mt19937 rng(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	std::uniform_int_distribution<size_t> dist(0, totalImages - 1);
	return dist(rng);
}

// Executes a full epoch consisting of multiple mini-batches and returns the average cost.
static float RunTrainingEpoch(Network& network,
                              const std::vector<std::vector<float>>& images,
                              const std::vector<std::vector<float>>& labels,
                              size_t& currentImageIdx,
                              const size_t printEveryNBatches,
                              const size_t batchSize,
                              float& learningRate)
{
	float totalCost = 0.0f;

	for (size_t j = 0; j < printEveryNBatches; j++)
	{
		for (size_t i = 0; i < batchSize; i++)
		{
			float c = network.BackPropagate(images[currentImageIdx], labels[currentImageIdx]);
			totalCost += c;

			currentImageIdx = GetRandomImageIndex(images.size());
		}

		network.ConsumeDelta(learningRate);
		learningRate *= static_cast<float>(std::pow(0.9999999, batchSize));
	}

	return totalCost / static_cast<float>(batchSize * printEveryNBatches);
}
}

// ============================================================================
// Main Execution
// ============================================================================

int main()
{
	// Seed Random Number Generator
	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand(static_cast<uint32_t>(seed));

	// Load Input Data & Derive Checkpoint Filename
	const std::string targetImageFile = "Sarah.bmp";
	const std::string weightsFile = GetCheckpointFilename(targetImageFile);

	BMPParsedData data;
	ParseBMPData(targetImageFile.c_str(), data);
	// Alternative image loading: ParseBMPData("Sarah_large.bmp", data);

	const std::vector<std::vector<float>>& images = data.inputs;
	const std::vector<std::vector<float>>& labels = data.outputs;

	// Initialize Neural Network & Visualizer Window
	std::vector<size_t> layerDims{ 2,8,16,32,64,64,32,16, 3 };
	Network network{ layerDims };

	ImageWindow rendererWindow(data.width, data.height);

	// Load Weights Checkpoint (Validates dimensions vs current layerDims automatically)
	LoadCheckpoint(network, weightsFile);

	// Display Interactive Controls
	PrintControls();

	// Hyperparameters & Training State
	constexpr size_t batchSize = 32;
	constexpr size_t printEveryNBatches = 128;
	float learningRate = 0.25f;

	size_t currentImage = 0;
	bool liveUpdateWindow = true;

	// Main Training & UI Loop
	while (true)
	{
		// Pump Windows messages so the viewer window stays responsive
		rendererWindow.ProcessMessages();

		// Handle non-blocking user input
		if (!HandleUserAction(PollUserAction(), network, data, weightsFile, liveUpdateWindow))
		{
			break;
		}

		// Perform batch training step
		float cost = RunTrainingEpoch(network, images, labels, currentImage, printEveryNBatches, batchSize, learningRate);

		// Live viewer update
		if (liveUpdateWindow)
		{
			rendererWindow.Update(GenerateReconstructedImage(network, data.width, data.height));
		}

		// Report progress
		std::cout << "COST: " << cost << " LR: " << learningRate << '\n';
	}

	// Final Output Generation & Cleanup
	std::cout << "Generating final output image from network state...\n";

	std::vector<float> finalReconstructedImage = GenerateReconstructedImage(network, data.width, data.height);

	rendererWindow.Update(finalReconstructedImage);

	saveBMP("network_output.bmp", data.width, data.height, finalReconstructedImage);
	std::cout << "Successfully saved network_output.bmp!\n";

	SaveCheckpoint(network, weightsFile);
	std::cout << "Final " << weightsFile << " saved.\n";

	return 0;
}

