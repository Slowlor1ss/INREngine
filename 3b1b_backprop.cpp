#include "Network.h"
// This vexes me
#define NOMINMAX 
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
#include <ranges>
#include <string>
#include <vector>

#include "CudaManager.cuh"

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

#if _HAS_CXX23
    constexpr engineFloat K_PI = std::numbers::pi_v<engineFloat>;
#else
    constexpr engineFloat K_PI = static_cast<engineFloat>(3.141592653589793);
#endif

// ============================================================================
// Input & State Definitions
// ============================================================================

enum class UserAction : uint8_t {
	None,
	SaveWeights,
	ExportImage,
	ToggleViewer,
	SwapRenderMode,
	Quit,
};

namespace config
{
	inline bool use_gpu = true;
	inline bool benchmark_enabled = false;
	
	inline std::string target_image_file = "Training_Data/0064_x4.bmp";
	inline std::string output_path = "";
	inline std::string output_filename = target_image_file;

	inline engineFloat output_image_scale = 1.f;
	inline std::vector<size_t> custom_layer_dims = { 128, 128, 3 };
	inline std::vector<ActFunc::Base*> custom_activations = {
		ActFunc::DataBase::FindActFunc<ActFunc::Finer>(),
		ActFunc::DataBase::FindActFunc<ActFunc::Finer>(),
		ActFunc::DataBase::FindActFunc<ActFunc::None>()
	};

	//TODO-Lkrikilion: make a command like param for this like --render-mode or smth
	inline RenderMode render_mode = RenderMode::StandardRGB;

	inline bool use_positional_encoding = false;
	inline int pe_num_frequencies = 10; // Positional encode
	inline bool use_gaussian_pe = false;

	inline bool initial_live_update_state = true;

	// Hyperparameters & Training State
	// Note if we drop this below out thread count we will run singlethreaded (which should be fine)
	inline size_t batch_size = 256ull*256ull;//8192;//65536;//8192;//32;
	inline bool shuffle_pixel_batch = true; // TODO: either make this an input parameter or make this the default if batch size isnt == to image size
	inline size_t print_every_n_batches = 1000;
	//inline float initial_learning_rate = 0.0001f;
	inline engineFloat initial_learning_rate = 0.1f;//0.005f;//0.005f;//WIRE //0.000025f; Siren
	
	// Hyperparameter to balance how much the network cares about slopes vs colors
	// used by spatial gradient (use 0 to turn spatial gradient off)
	//inline engineFloat spatialLossWeight = 0.f; // TODO: make a utils file so we can use this
}

namespace
{
// ============================================================================
// Helper Functions
// ============================================================================
	
// MAKE SURE TO ADD TO THE PRINT AT THE BOTTOM WHEN ADDING ARGS!
static int ParseCommandLine(const int argc, char** argv)
{
	for (int i = 1; i < argc; ++i) // Start at 1 because argv[0] is the program name
	{
		std::string arg = argv[i];
		if (arg == "--help" || arg == "--h" || arg == "-h" || arg == "help")
		{
			std::cout << std::format(
				"Usage:\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"  {:<14} | {}\n"
				"==================================================================================================\n",
				"--i",			"Set input filename",
				"--o",			"Set output path (can specify file as well e.g. weights_biases.csv)",	
				"--gpu",		"Enable CUDA",
				"--benchmark",	"Enable Benchmarking",
				"--layers",		"Set the layers e.g. --layers 128 128 3",
				"--act",		"Set the activation function(s) e.g. --act Wire Siren None",
				"",				"Valid options are: " + ActFunc::DataBase::GetAllActNames(),
				"--set-pe 0/ 1", "Enable positional encoding",
				"--freq",		"Set positional encoding Frequencies",
				"--set-gaussian-pe 0/1", "Set Gaussian Positional Encoding (does NOT use --freq)",
				"--batch",		"Set batch Size",
				"--lr",			"Set the learning Rate",
				"--set-live 0/1", "Disable live viewer on start; Note: this can be re-enabled during runtime using 'v'"
			);
			return 0;
		}
		else if (arg == "--gpu" && i + 1 < argc)
		{
			config::use_gpu = std::stoi(argv[++i]);
		}
		else if (arg == "--benchmark" && i + 1 < argc)
		{
			config::benchmark_enabled = std::stoi(argv[++i]);
		}
		else if (arg == "--layers") 
		{
			config::custom_layer_dims.clear();
			// Keep reading the next arguments as long as they don't start with '-'
			while (i + 1 < argc && argv[i + 1][0] != '-') {
				try {
					// Convert the string argument to an unsigned long integer (size_t)
					config::custom_layer_dims.push_back(std::stoul(argv[i + 1]));
				} catch (const std::exception& e) {
					std::cerr << "Error parsing layer dimension: " << argv[i + 1] << "Error: " << e.what() << "\n";
					// TODO maybe return -1;
				}
				i++; // Advance the loop
			}
		}
		else if (arg == "--act")
		{
			config::custom_activations.clear();
			// Keep reading the next arguments as long as they don't start with '-'
			while (i + 1 < argc && argv[i + 1][0] != '-') {
				try {
					// Convert the string argument to an unsigned long integer (size_t)
					config::custom_activations.push_back( ActFunc::DataBase::FindActFunc(argv[i + 1]) );
				} catch (const std::exception& e) {
					std::cerr << "Error parsing layer activation: " << argv[i + 1] << "Error: " << e.what() << "\n";
				}
				i++; // Advance the loop
			}
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
		else if (arg == "--set-gaussian-pe" && i + 1 < argc)
		{
			config::use_gaussian_pe = std::stoi(argv[++i]);
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
			//std::exit(127);
		}
	}
	
	// C++23 Native Range Formatting doesnt work for some dammed reason
	// std::string customLayersStr = config::custom_layer_dims.empty() 
	// 	? "Default" 
	// 	: std::format("{}", config::custom_layer_dims);
	//
	// std::string customActsStr = config::custom_activations.empty() 
	// 		? "Default" 
	// 		: std::format("{}", config::custom_activations | std::views::transform([](ActFunc::Base* act) { return act->GetName(); }));

	std::string customLayersStr = "Default";
	if (!config::custom_layer_dims.empty()) {
		customLayersStr = "[";
		customLayersStr += "Ninput, "; // TODO: replace this with the actual input size but its depending on what PE or any
		for (size_t i = 0; i < config::custom_layer_dims.size(); ++i) {
			customLayersStr += std::to_string(config::custom_layer_dims[i]);
			if (i < config::custom_layer_dims.size() - 1) customLayersStr += ", ";
		}
		customLayersStr += "]";
	}
	
	std::string customActsStr = "Default";
	if (!config::custom_activations.empty()) {
		customActsStr = "[";
		customActsStr += ActFunc::None().GetName() + ", ";
		for (size_t i = 0; i < config::custom_activations.size(); ++i) {
			customActsStr += config::custom_activations[i]->GetName();
			if (i < config::custom_activations.size() - 1) customActsStr += ", ";
		}
		customActsStr += "]";
	}
	
	// Print the final configuration state
	std::cout << std::format(
	   "=== Launch Configuration ===\n"
	   " Input file          : {}\n"
	   " Output path         : {}\n"
	   " Use GPU             : {}\n"
	   " Benchmark           : {}\n"
	   " Topology (Layers)   : {}\n"
	   " Activations         : {}\n"
	   " Positional Encoding : {}\n"
	   " PE Frequencies      : {}\n"
	   " Gaussian PE         : {}\n"
	   " Batch Size          : {}\n"
	   " Learning Rate       : {:.4f}\n"
	   " Live Viewer         : {}\n"
	   "============================\n",
	   config::target_image_file,
	   (fs::path(config::output_path) / config::output_filename).string(),
	   config::use_gpu ? "ON" : "OFF",
	   config::benchmark_enabled ? "ON" : "OFF",
	   customLayersStr,
	   customActsStr,
	   config::use_positional_encoding ? "ON" : "OFF",
	   config::use_positional_encoding ? std::to_string(config::pe_num_frequencies) : "DISABLED",
	   config::use_gaussian_pe ? "ON" : "OFF",
	   config::batch_size,
	   config::initial_learning_rate,
	   config::initial_live_update_state ? "ON" : "OFF"
	);
	
	return 1;
}

// TODO: merge the 2 function below or something this is bad but we need one for the imagedataset and another for the coormapper
static ImageUtils::SpatialData PositionalEncodeWithDerivatives(engineFloat x, engineFloat y, int numFrequencies) 
{
    ImageUtils::SpatialData result;
    const size_t size = static_cast<size_t>(numFrequencies) * 4;
    result.values.reserve(size);
    result.gradX.reserve(size);
    result.gradY.reserve(size);

    for (int i = 0; i < numFrequencies; ++i) {
        const engineFloat freq = std::powf(2.f, float(i)) * K_PI;
        const engineFloat weight = 1.0f - (static_cast<float>(i) / static_cast<float>(numFrequencies));

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

// Helper to expand a coordinate (x, y) into multiple frequency bands with decay
static std::vector<engineFloat> PositionalEncode(engineFloat x, engineFloat y, int numFrequencies) {
	std::vector<engineFloat> encoded;
	encoded.reserve(static_cast<size_t>(numFrequencies) * 4);

	for (int i = 0; i < numFrequencies; ++i) {
		const engineFloat freq = std::pow(2.0f, static_cast<engineFloat>(i)) * K_PI;
		const engineFloat weight = 1.0f - (static_cast<engineFloat>(i) / static_cast<engineFloat>(numFrequencies));
    
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

	// Get filename without extension (aka: stem)
	std::string stem = fs::path(imageFilename).stem().string();
	std::string filename = "weights_biases_" + stem + ".csv";

	// If a base path is provided, modify the filename variable
	if (!outbasePath.empty())
	{
		filename = (fs::path(outbasePath) / filename).string();
	}
	
	return filename; 
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
		<< " [D]     - Swap render mode (Used for debugging)\n"
		<< " [E]     - Save image to disk\n"
		<< " [V]     - Toggle live viewer window\n\n";
}

// Polls non-blocking keyboard input buffer and maps to UserAction.
static UserAction PollUserAction()
{
	if (!_kbhit()) return UserAction::None;

	switch (int ch = _getch())
	{
		case 'q': case 'Q': case 27: // ESC
			return UserAction::Quit;
		case 'w': case 'W':
			return UserAction::SaveWeights;
		case 'e': case 'E':
			return UserAction::ExportImage;
		case 'v': case 'V':
			return UserAction::ToggleViewer;
		case 'd': case 'D':
			return UserAction::SwapRenderMode;
		default:
			return UserAction::None;
	}
}

// Handles user actions outside the main training loop; returns false to break loop.
static bool HandleUserAction(const UserAction action, Network& network, const BMPParsedData& data,
                             const std::string& weightsFile, bool& liveUpdateWindow,
                             const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& mapper,
                             TrainingThreadPool& threadPool)
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
			const std::vector<engineFloat> reconstructedImage = GenerateReconstructedImage(network, int(data.width * config::output_image_scale), int(data.height * config::output_image_scale), mapper, config::render_mode, threadPool);
			saveBMP("network_output.bmp", int(data.width*config::output_image_scale), int(data.height*config::output_image_scale), reconstructedImage);
			std::cout << "Successfully saved network_output.bmp!\n";
			break;
		}
		
		case UserAction::ToggleViewer:
			liveUpdateWindow = !liveUpdateWindow;
			std::cout << (liveUpdateWindow ? "Live viewer window ENABLED\n" : "Live viewer window DISABLED\n");
			break;

		case UserAction::SwapRenderMode:
			config::render_mode = (RenderMode)(((int)config::render_mode + 1) % (int)RenderMode::Last);
			std::cout << "Updated render mode!\n"; // Im not making an enum to sting >:(
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
// static engineFloat RunTrainingEpoch(Network& network,
//                               const std::vector<std::vector<engineFloat>>& images,
//                               const std::vector<std::vector<engineFloat>>& labels,
//                               size_t& currentImageIdx,
//                               const size_t printEveryNBatches,
//                               const size_t batchSize,
//                               engineFloat& learningRate)
// {
// 	engineFloat totalCost = 0.0f;
//
// 	for (size_t j = 0; j < printEveryNBatches; j++)
// 	{
// 		for (size_t i = 0; i < batchSize; i++)
// 		{
// 			engineFloat c = network.BackPropagate(images[currentImageIdx], labels[currentImageIdx]);
// 			totalCost += c;
//
// 			currentImageIdx = GetRandomImageIndex(images.size());
// 		}
//
// 		network.ConsumeDelta(learningRate);
// 		learningRate *= static_cast<engineFloat>(std::pow(0.9999999, batchSize));
// 	}
//
// 	return totalCost / static_cast<engineFloat>(batchSize * printEveryNBatches);
// }
}

#include "NeuralImageRecreator.hpp"

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv)
{
	// Forces cuBLAS to boot up immediately rather then upon first use
	CudaManager::GetInstance(); 
	cudaError_t initErr = cudaGetLastError();
	
	// Parse the command line arguments right at startup
	if ( int ret = ParseCommandLine(argc, argv); ret != 1 )
		return ret; // Returns 0 on help, -1 if we went very wrong
	
	// Seed Random Number Generator
	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand(static_cast<uint32_t>(seed));

	NeuralImageRecreator();

	return 0;
}

