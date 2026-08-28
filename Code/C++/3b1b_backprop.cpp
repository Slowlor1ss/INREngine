#include "Network.h"
#include "WindowRenderer.h"
#include "ImageGenerator.h"
#include "CostFuncDataBase.h"
#include "Gemini/ImageParser.h"

#include "Config.h"
#include "NeuralImageRecreator.h"
#include "ArgParser.h"

#include <chrono>
#include <cmath>
#include <conio.h> // For _kbhit() and _getch()
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

#include "../cuda/CudaManager.cuh"
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

// TODO: replace this aswell with some automated ArgParser function
// Prints the final resolved configuration once all arguments have been applied.
// Runs as ArgParser's post-parse hook, so it's skipped entirely when --help is requested.
static void PrintLaunchConfig()
{
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
	   " HD reference path   : {}\n"
	   " Use GPU             : {}\n"
	   " Benchmark           : {}\n"
	   " Topology (Layers)   : {}\n"
	   " Activations         : {}\n"
	   " Positional Encoding : {}\n"
	   " PE Frequencies      : {}\n"
	   " Gaussian PE         : {}\n"
	   " Batch Size          : {}\n"
	   " Learning Rate       : {:.4f}\n"
	   " Resize Scale        : {}\n"
	   " Live Viewer         : {}\n"
	   " Denoise Jitter      : {}\n"
	   "============================\n",
	   config::target_image_file,
	   (fs::path(config::output_path) / config::output_filename).string(),
	   config::hd_image_file,
	   config::use_gpu ? "ON" : "OFF",
	   config::benchmark_enabled ? "ON" : "OFF",
	   customLayersStr,
	   customActsStr,
	   config::use_positional_encoding ? "ON" : "OFF",
	   config::use_positional_encoding ? std::to_string(config::pe_num_frequencies) : "DISABLED",
	   config::use_gaussian_pe ? "ON" : "OFF",
	   config::batch_size,
	   config::initial_learning_rate,
	   config::output_image_scale,
	   config::initial_live_update_state ? "ON" : "OFF",
	   config::use_denoise_jitter ? "ON" : "OFF"
	);
}
	
// Hyperparameters arguments
static void ParseAtivationConfig(ArgParser& parser)
{
	parser.AddArgument<float>("--Siren_w0", "", SharedAct::Config::Siren_w0);
	
	parser.AddArgument<float>("--Wire_w0", "", SharedAct::Config::Wire_w0);
	parser.AddArgument<float>("--Wire_s", "", SharedAct::Config::Wire_s);
	parser.AddHelpNote("We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation");
	
	parser.AddArgument<float>("--LeakyReLU_slope", "", SharedAct::Config::LeakyReLU_slope);
	
	parser.AddArgument<float>("--Finer_w0", "", SharedAct::Config::Finer_w0);
	parser.AddArgument<float>("--Finer_bias_k", "Bias init range for FINER, "
					"this is what actually gives it its extra", SharedAct::Config::Finer_bias_k);
	parser.AddHelpNote("frequency range over SIREN");
}
	
static void ParseINRConfig(ArgParser& parser)
{
	parser.AddAction("--i", "Set input filename",
	[](int& i, int argc, char** argv)
	{
		if (i + 1 >= argc) return;
		std::string val = argv[++i];
		// Make sure we always have a output path set, or if its set to be the same as input, update it alongside
		if (config::output_filename.empty() || config::output_filename == config::target_image_file)
		{
			config::output_filename = val;
			config::target_image_file = val;
		}
		else
		{
			config::target_image_file = val;
		}
	});

	parser.AddAction("--o", "Set output path (can specify file as well e.g. weights_biases.csv)",
		[](int& i, int argc, char** argv)
		{
			if (i + 1 >= argc) return;
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
		});
	
	parser.AddArgument<std::string>("--HDin", "Set path to HD reference image", config::hd_image_file);
	parser.AddArgument<bool>("--gpu", "Enable CUDA", config::use_gpu);
	parser.AddArgument<bool>("--benchmark", "Enable Benchmarking", config::benchmark_enabled);
	parser.AddMultiValue<size_t>("--layers", "Set the layers e.g. --layers 128 128 3", config::custom_layer_dims);

	parser.AddMultiValue<ActFunc::Base*>("--act", "Set the activation function(s) e.g. --act Wire Siren None", config::custom_activations,
		[](const std::string& s) { return ActFunc::DataBase::FindActFunc(s.c_str()); });
	parser.AddHelpNote("Valid options are: " + ActFunc::DataBase::GetAllActNames());

	parser.AddArgument<bool>("--set-pe", "0/1 Enable positional encoding", config::use_positional_encoding);
	parser.AddArgument<int>("--freq", "Set positional encoding Frequencies", config::pe_num_frequencies);
	parser.AddArgument<bool>("--set-gaussian-pe", "0/1 Set Gaussian Positional Encoding (does NOT use --freq)", config::use_gaussian_pe);
	parser.AddArgument<size_t>("--batch", "Set batch Size", config::batch_size);
	parser.AddArgument<float>("--lr", "Set the learning Rate", config::initial_learning_rate);
	parser.AddArgument<float>("--scale", "Set the resize scale", config::output_image_scale);
	parser.AddArgument<bool>("--set-live", "0/1 Disable live viewer on start; Note: this can be re-enabled during runtime using 'v'", config::initial_live_update_state);
	parser.AddArgument<bool>("--denoise", "0/1 Weather or not we use denoise jitter", config::use_denoise_jitter);

}
	
// MAKE SURE TO ADD TO THE PRINT AT THE BOTTOM WHEN ADDING ARGS!
// Adding a new flag is now just one AddArgument/AddMultiValue/AddAction call below -
// help text, value parsing, and index-advancing are all handled by ArgParser itself.
static int ParseCommandLine(const int argc, char** argv)
{
	ArgParser parser(R"(A (no libraries) custom made C++/CUDA framework for Implicit Neural Representations (INRs). Using CUDA to run the network on the GPU.
)");
	
	ParseINRConfig(parser);
	ParseAtivationConfig(parser);
	
	parser.SetPostParseHook(PrintLaunchConfig);

	return parser.Parse(argc, argv);
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