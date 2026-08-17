#include "Network.h"
#include "WindowRenderer.h"
#include "ImageGenerator.h"
#include "CostFuncDataBase.h"
#include "Gemini/BMPParser.h"

#include "Config.h"
#include "NeuralImageRecreator.h"

#include <chrono>
#include <cmath>
#include <conio.h> // For _kbhit() and _getch()
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

#include "CudaManager.cuh"
#include "GpuNetwork.h"

namespace fs = std::filesystem;

// TODO: don't see the point for the database, just use them directly? to have a single instance? does that matter?
// seems to be to use the inheritance more easily >> convert to template instead?
// 
// TODO: Relu needs HE initialization: make it so that happens automatically when selecting relu, and not when selecting sigmoid for a layer.
// TODO: serialization
// TODO: better interface / frontend.

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
}

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv)
{
	// Forces cublas to boot up immediately rather then upon first use
	CudaManager::GetInstance(); 
	cudaError_t initErr = cudaGetLastError();
    if ( initErr != cudaSuccess ) {
        printf( "\n[INIT ERROR] Cuda Init Error: %s\n", cudaGetErrorString( initErr ) );
        __debugbreak();
    }
	
	// Parse the command line arguments right at startup
	if ( int ret = ParseCommandLine(argc, argv); ret != 1 )
		return ret; // Returns 0 on help, -1 if we went very wrong
	
	// Seed Random Number Generator
	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand(static_cast<uint32_t>(seed));

	NeuralImageRecreator recreator;
	recreator.Run();

	return 0;
}