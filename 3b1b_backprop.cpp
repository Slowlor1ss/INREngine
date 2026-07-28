#include "Network.h"
#include "WindowRenderer.h"
#include "ImageGenerator.h"
#include "CostFuncDataBase.h"
#include "Gemini/BMPParser.h"

// Alternative dataset readers (available for switching modes)
#include "Gemini/MNISTReader.h"
#include "Gemini/CustomFileReader.h"

#include <chrono>
#include <cmath>
#include <conio.h> // For _kbhit() and _getch()
#include <filesystem>
#include <fstream>
#include <iostream>
#if _HAS_CXX23
	#include <numbers>
#endif
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

namespace config
{
	inline std::string target_image_file = "laurens.bmp";
	inline std::string output_path = "";
	inline std::string output_filename = target_image_file;
	
	inline bool use_positional_encoding = true;
	inline int pe_num_frequencies = 7; // Positional encode

	inline bool initial_live_update_state = true;

	// Hyperparameters & Training State
	inline size_t batch_size = 32;
	inline size_t print_every_n_batches = 1024;
	inline float initial_learning_rate = 0.25f;
}

namespace
{
// ============================================================================
// Helper Functions
// ============================================================================
	
// MAKE SURE TO ADD TO THE PRINT AT THE BOTTOM WHEN ADDING ARGS!
static void ParseCommandLine(const int argc, char** argv)
{
	for (int i = 1; i < argc; ++i) // Start at 1 because argv[0] is the program name
	{
		std::string arg = argv[i];
		if (arg == "--h" || arg == "-h" || arg == "help")
		{
			std::cout << std::format(
				"Usage:\n"
				"  {:<10} | {}\n"
				"  {:<10} | {}\n"
				"  {:<10} | {}\n"
				"  {:<10} | {}\n"
				"  {:<10} | {}\n"
				"==================================================================================================\n",
				"--i",			"Set input filename",
				"--o",			"Set output path (can specify file aswell e.g. weights_biases.csv)",	
				"--set-pe 0/ 1", "Enable positional encoding",
				"--freq",		"Set positional encoding Frequencies",
				"--batch",		"Set batch Size",
				"--lr",			"Set the learning Rate",
				"--set-live 0/1", "Disable live viewer on start; Note: this can be re-enabled during runtime using 'v'"
			);
		}
		else if (arg == "--i")
		{
			// Make sure we always have a output path set or if its set to be the same ans input updat e it alongside
			if (config::output_filename.empty() || config::output_filename == config::target_image_file)
			{
				config::output_filename = argv[++i];
				config::target_image_file = config::output_filename;
			}
			else
			{
				config::target_image_file = argv[++i];
			}
		}
		else if (arg == "--o" && i + 1 < argc)
		{
			fs::path providedPath(argv[++i]);

			// If the path has an extension it's a file
			if (providedPath.has_extension())
			{
				// .parent_path() grabs everything BEFORE the filename (can be empty)
				config::output_path = providedPath.parent_path().string();
             
				// .filename() grabs just the file and its extension
				config::output_filename = providedPath.filename().string();
			}
			else
			{
				// If there's no extension, assume it's just a directory
				config::output_path = providedPath.string();
				config::output_filename = config::target_image_file;
			}
		}
		else if (arg == "--set-pe")
		{
			config::use_positional_encoding = std::stoi(argv[++i]);
		}
		else if (arg == "--freq" && i + 1 < argc)
		{
			config::pe_num_frequencies = std::stoi(argv[++i]); // Read next arg as int
		}
		else if (arg == "--batch" && i + 1 < argc)
		{
			config::batch_size = std::stoull(argv[++i]); // Read next arg as size_t
		}
		else if (arg == "--lr" && i + 1 < argc)
		{
			config::initial_learning_rate = std::stof(argv[++i]); // Read next arg as float
		}
		else if (arg == "--set-live")
		{
			config::initial_live_update_state = std::stoi(argv[++i]);
		}
		else
		{
			std::cout << "Unknown or incomplete argument: " << arg << "\n";
		}
	}
	
	// Print the final configuration state after parsing is complete
	std::cout << std::format(
		"=== Launch Configuration ===\n"
		" Input file		  : {}\n"
		" Output path		  : {}\n"
		" Positional Encoding : {}\n"
		" PE Frequencies      : {}\n"
		" Batch Size          : {}\n"
		" Learning Rate       : {:.4f}\n"
		" Live Viewer         : {}\n"
		"============================\n",
		config::target_image_file,
		(fs::path(config::output_path) / config::output_filename).string(),
		config::use_positional_encoding ? "ON" : "OFF",
		config::use_positional_encoding ? std::to_string(config::pe_num_frequencies) : "DISABLED",
		config::batch_size,
		config::initial_learning_rate,
		config::initial_live_update_state ? "ON" : "OFF"
	);
}
	
// Helper to expand a coordinate (x, y) into multiple frequency bands with decay
static std::vector<float> PositionalEncode(float x, float y, int numFrequencies) {
	std::vector<float> encoded;
	encoded.reserve(static_cast<size_t>(numFrequencies) * 4);

#if _HAS_CXX23
	constexpr float PI = std::numbers::pi_v<float>; // Finnaly standard PI
#else
	constexpr float PI = static_cast<float>(3.141592653589793);
#endif

	for (int i = 0; i < numFrequencies; ++i) {
		const float freq = std::powf(2.0f, float(i)) * PI;
		const float weight = 1.0f - (static_cast<float>(i) / static_cast<float>(numFrequencies));
    
		encoded.push_back(std::sin(x * freq) * weight);
		encoded.push_back(std::cos(x * freq) * weight);
    
		encoded.push_back(std::sin(y * freq) * weight);
		encoded.push_back(std::cos(y * freq) * weight);
	}
	return encoded;
}
	
// Generates a checkpoint filename based on the input image filename.
// Example: "Sarah.bmp" -> "weights_biases_Sarah.csv"
static std::string GetCheckpointFilename(const std::string& imageFilename, const std::string& outbasePath = "")
{
	namespace fs = std::filesystem;

	// Get filename withouth extention (aka: stem)
	std::string stem = fs::path(imageFilename).stem().string();
	std::string filename = "weights_biases_" + stem + ".csv";

	// If no base path is provided, just return the filename
	if (outbasePath.empty())
	{
		return filename;
	}

	// The '/' operator safely joins paths, automatically adding slashes if needed!
	return (fs::path(outbasePath) / filename).string();
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
static bool HandleUserAction(const UserAction action, Network& network, const BMPParsedData& data,
                             const std::string& weightsFile, bool& liveUpdateWindow,
                             const std::function<std::vector<float>(float, float)>& mapper)
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
			std::vector<float> reconstructedImage = GenerateReconstructedImage(network, data.width, data.height, mapper);
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

#include "NeuralImageRecreator.hpp"

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv)
{
	// Parse the command line arguments right at startup
	ParseCommandLine(argc, argv);
	
	// Seed Random Number Generator
	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand(static_cast<uint32_t>(seed));

	NeuralImageRecreator();

	return 0;
}

