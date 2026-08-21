#include "CheckpointUtils.h"
#include "Config.h"

#include <filesystem>
#include <fstream>
#include <iostream>

std::string GetCheckpointFilename(const std::string& imageFilename, const std::string& outbasePath)
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

void LoadCheckpoint(Network& network, const std::string& filename)
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

// Horrible explains absolutely nothing but i find it mildly humorous :)
inline void _You_see_this_error_because_you_did_something_very_very_wrong() noexcept {}
void SaveCheckpoint(Network& network, GpuNetwork* gpuNet, const std::string& filename)
{
	if (config::use_gpu) {
		// This should in theory not be possible but if we do hit it maybe ill get a chuckle out of it
		if ( gpuNet == nullptr) throw _You_see_this_error_because_you_did_something_very_very_wrong; 
		
		gpuNet->DownloadParametersToCPU(network);
	}

	std::ofstream outFile{ filename };
	if (outFile.is_open())
	{
		network.Serialize(outFile);
		std::cout << "Weights saved to " << filename << "!\n";
	}
}
