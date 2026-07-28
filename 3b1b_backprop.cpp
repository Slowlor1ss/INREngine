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
#include <thread>

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

enum class UserAction {
	None,
	SaveWeights,
	ToggleViewer,
	Quit
};

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
		case 'v': case 'V':
			return UserAction::ToggleViewer;
		default:
			return UserAction::None;
	}
}

// Handles user actions outside the main training loop; returns false to break loop.
static bool HandleUserAction(UserAction action, Network& network, const std::string& weightsFile, bool& liveUpdateWindow)
{
	switch (action)
	{
		case UserAction::Quit:
			std::cout << "\n[Interrupt Received] Stopping training...\n";
			return false;

		case UserAction::SaveWeights:
			SaveCheckpoint(network, weightsFile);
			break;
		case UserAction::ToggleViewer:
			liveUpdateWindow = !liveUpdateWindow;
			std::cout << (liveUpdateWindow ? "Live viewer window ENABLED\n" : "Live viewer window DISABLED\n");
			break;

		case UserAction::None:
		default:
			break;
	}

	return true;
}

// Generates a 32-bit random index to correctly sample datasets larger than RAND_MAX (32,767).
static size_t GetRandomImageIndex(size_t totalImages)
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
                              size_t printEveryNBatches,
                              size_t batchSize,
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

// ============================================================================
// Main Execution
// ============================================================================

static void LoadMNISTData(std::vector<std::vector<float>>& trainImages,
	std::vector<std::vector<float>>& trainLabels,
	std::vector<std::vector<float>>& testImages,
	std::vector<std::vector<float>>& testLabels)
{
	trainImages = read_mnist_images("DATA/train-images-idx3-ubyte");
	trainLabels = read_mnist_labels("DATA/train-labels-idx1-ubyte");
	testImages = read_mnist_images("DATA/t10k-images-idx3-ubyte");
	testLabels = read_mnist_labels("DATA/t10k-labels-idx1-ubyte");
}

int main()
{
	// Set to true to explore the Latent Space!
	bool isTestMode = true;

	if (isTestMode)
	{
		std::cout << "==================================\n\n";

		std::vector<Network::LayerInfo> layerDims{
			{2, ActFunc::DataBase::FindActFunc<ActFunc::Empty>()},
			{16, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{128, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{256, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{28 * 28, ActFunc::DataBase::FindActFunc<ActFunc::Sigmoid>()} };

		Network network{ layerDims, CostFunc::DataBase::FindCostFunc<CostFunc::L1>() };

		LoadCheckpoint(network, "mnist_decoder.csv");

		int scale = 10;
		ImageWindow rendererWindow(28 * scale, 28 * scale);

		float latent_x = 0.0f;
		float latent_y = 0.0f;
		float step = 1.0f; // How fast you move through the space

		std::cout << "Controls:\n [W/A/S/D] - Move in Latent Space\n [Q/ESC]   - Quit\n\n";
		std::cout << "Current Coordinate: (" << latent_x << ", " << latent_y << ")\r";

		while (true)
		{
			rendererWindow.ProcessMessages();

			bool changed = true; // Force render on first frame
			if (_kbhit())
			{
				int ch = _getch();
				switch (ch)
				{
				case 'w': case 'W': latent_y += step; changed = true; break;
				case 's': case 'S': latent_y -= step; changed = true; break;
				case 'a': case 'A': latent_x -= step; changed = true; break;
				case 'd': case 'D': latent_x += step; changed = true; break;
				case 'q': case 'Q': case 27: return 0;
				}
			}

			if (changed)
			{
				// The \r and trailing spaces keep the console clean on a single line
				std::cout << "Current Coordinate: (" << latent_x << ", " << latent_y << ")          \r";

				// Run the forward pass and update the screen
				rendererWindow.Update(GenerateImageFromLatent(network, latent_x, latent_y , scale));
			}

			// Small sleep to prevent frying your CPU (roughly 60 FPS)
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}
		return 0;

	}
	//else
	//{
	//	// 1. Seed Random Number Generator
	//	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	//	srand(static_cast<uint32_t>(seed));

	//	// 2. Load Input Data & Derive Checkpoint Filename
	//	const std::string weightsFile = GetCheckpointFilename("mnist");

	//	std::vector<std::vector<float>> images, labels, testImages, testLabels;


	//	LoadMNISTData(images, labels, testImages, testLabels);

	//	labels = images;
	//	testLabels = testImages;

	//	// 3. Initialize Neural Network & Visualizer Window
	//	std::vector<size_t> layerDims{ 28 * 28,256,128,16,2,16,128,256,28 * 28 };
	//	Network network{ layerDims, CostFunc::DataBase::FindCostFunc<CostFunc::L1>() };

	//	ImageWindow rendererWindow(28 * 2, 28);

	//	// 4. Load Weights Checkpoint (Validates dimensions vs current layerDims automatically)
	//	LoadCheckpoint(network, weightsFile);

	//	// 5. Display Interactive Controls
	//	PrintControls();

	//	// 6. Hyperparameters & Training State
	//	const size_t batchSize = 128;
	//	const size_t printEveryNBatches = 2;
	//	float learningRate = 0.0001;

	//	size_t currentImage = 0;
	//	bool liveUpdateWindow = true;

	//	// 7. Main Training & UI Loop
	//	while (true)
	//	{
	//		// Pump Windows messages so the viewer window stays responsive
	//		rendererWindow.ProcessMessages();

	//		// Handle non-blocking user input
	//		if (!HandleUserAction(PollUserAction(), network, weightsFile, liveUpdateWindow))
	//		{
	//			break;
	//		}

	//		// Perform batch training step
	//		float cost = RunTrainingEpoch(network, images, labels, currentImage, printEveryNBatches, batchSize, learningRate);

	//		// Live viewer update
	//		if (liveUpdateWindow)
	//		{
	//			rendererWindow.Update(GenerateReconstructedImage(network, images[GetRandomImageIndex(images.size())]));
	//		}

	//		// Report progress
	//		std::cout << "COST: " << cost << " LR: " << learningRate << '\n';
	//	}

	//	// 8. Final Output Generation & Cleanup
	//	std::cout << "Generating final output image from network state...\n";

	//	SaveCheckpoint(network, weightsFile);
	//	std::cout << "Final " << weightsFile << " saved.\n";
	//}
	return 0;
}

