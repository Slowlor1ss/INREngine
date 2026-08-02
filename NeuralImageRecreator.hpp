#include "TrainingThreadPool.h"
#include <numeric> // Put this at the very top of your file
#include <random>

static float RunGradientGuidedTrainingEpoch(Network& network,
                                            const std::vector<ImageUtils::SpatialData>& inputs,
                                            const std::vector<ImageUtils::SpatialData>& targets,
                                            size_t& currentImageIdx,
                                            const size_t printEveryNBatches,
                                            const size_t batchSize,
                                            float& learningRate,
                                            TrainingThreadPool& threadPool)
{
    float totalCost = 0.0f;

    for (size_t j = 0; j < printEveryNBatches; j++)
    {
        // 1. Run the entire batch across all cores instantly
        float batchCost = threadPool.RunBatch(inputs, targets, currentImageIdx, batchSize);
        totalCost += (batchCost / static_cast<float>(batchSize));
        
        // 2. Apply the averaged batch weights via Adam
        network.ConsumeDelta(learningRate);
        
        // 3. Move forward in the dataset (or pick random)
        currentImageIdx = GetRandomImageIndex(inputs.size());
        
        // Decay learning rate
		// Comented out for now as were now using Adam
        //learningRate *= static_cast<float>(std::pow(0.9999999, batchSize));
    }

    return totalCost / static_cast<float>(printEveryNBatches);
}

void NeuralImageRecreator()
{
	// Load Input Data & Derive Checkpoint Filename
	const std::string weightsFile = GetCheckpointFilename(config::output_filename, config::output_path);

	BMPParsedData data;
	ParseBMPData(config::target_image_file.c_str(), data);

	// Setup our coordinate mapper lambda for the ImageGenerator
	std::function<ImageUtils::SpatialData(float, float)> coordMapper = nullptr;
	if (config::use_positional_encoding) 
	{
	    coordMapper = [freqs = config::pe_num_frequencies](float x, float y) {
	        // No longer scaling by 2.0/w! Keep it pure.
	        return PositionalEncodeWithDerivatives(x, y, freqs); 
	    };
	}
	else 
	{
	    coordMapper = [](float x, float y) {
	        ImageUtils::SpatialData d;
	        d.values = { x, y };
	        d.gradX = { 1.0f, 0.0f }; // Pure normalized derivative
	        d.gradY = { 0.0f, 1.0f }; // Pure normalized derivative
	        return d;
	    };
	}

	// 1. Generate Target Outputs (RGB + Spatial Edges)
	std::vector<ImageUtils::SpatialData> targetSpatialData = ImageUtils::GenerateGradientTargets(data.outputs, data.width, data.height);

	// 2. Generate Inputs (Coordinates/PE + Spatial Slopes)
	std::vector<ImageUtils::SpatialData> inputSpatialData;
	inputSpatialData.reserve(data.inputs.size());
	for (const auto& rawInput : data.inputs) {
		inputSpatialData.push_back(coordMapper(rawInput[0], rawInput[1]));
	}

	///
	// TODO: resize vectors in if statement to not wase memory do this soon or will forget
	// Create new vectors to hold the randomized data
	std::vector<ImageUtils::SpatialData> shuffledInputs(inputSpatialData.size());
	std::vector<ImageUtils::SpatialData> shuffledTargets(targetSpatialData.size());
	if( config::shuffle_pixel_batch )
	{
		std::cout << "Shuffling dataset for random pixel batching...\n";

		// Create an array of indices [0, 1, 2, ... 65535]
		std::vector<size_t> indices(inputSpatialData.size());
		std::iota(indices.begin(), indices.end(), 0);

		// Shuffle the indices
		std::mt19937 g(1337); // Fixed seed for consistency
		std::shuffle(indices.begin(), indices.end(), g);


		for(size_t i = 0; i < indices.size(); ++i) {
		    shuffledInputs[i] = inputSpatialData[indices[i]];
		    shuffledTargets[i] = targetSpatialData[indices[i]];
		}
	}
	///

	// Initialize Neural Network & Visualizer Window
	size_t inputLayerSize = config::use_positional_encoding ? (config::pe_num_frequencies * 4) : 2;
	//std::vector<size_t> layerDims{ inputLayerSize, 8, 16, 32, 64, 64, 32, 16, 3 };
	//std::vector<size_t> layerDims{ inputLayerSize, 64, 64, 64, 64, 3 };
	//std::vector<size_t> layerDims{ inputLayerSize, 128, 128, 3 };
	//std::vector<size_t> layerDims{ inputLayerSize, 8, 16, 32, 16, 8, 3 };
	std::vector<size_t> layerDims;
	layerDims.push_back(inputLayerSize);
	if (!config::custom_layer_dims.empty()) {
		layerDims.insert(layerDims.end(), config::custom_layer_dims.begin(), config::custom_layer_dims.end());
	} else {
		//layerDims.insert(layerDims.end(), { 64, 64, 64, 64, 3 });
		layerDims.insert(layerDims.end(), { 256, 256, 256, 256, 3 });

	}
	
	// Pass the activation functions dynamically!
	Network network{ 
		layerDims, 
		ActFunc::DataBase::FindActFunc<ActFunc::Siren>(),
		// TODO: look in to this more maybe just use sigmoid as its basically the same or none as its more truthfully ig
		// and the docmentation says to just use a linear or sine https://deepwiki.com/vsitzmann/siren/2-siren-architecture#sinelayer-and-network-structure
		ActFunc::DataBase::FindActFunc<ActFunc::Sigmoid>() 
	};

	// Create the thread pool using your Ryzen's hardware concurrency (usually 16 threads for the 4800H)
	unsigned int numThreads = std::thread::hardware_concurrency();
	if (numThreads == 0) numThreads = 8;
	TrainingThreadPool threadPool(numThreads, network);

	ImageWindow rendererWindow(data.width * config::output_image_scale, data.height * config::output_image_scale);

	// Load Weights Checkpoint (Validates dimensions vs current layerDims automatically)
	// TODO: REENABLE ONCE I FIXED THIS DAMMED BUG
	LoadCheckpoint(network, weightsFile);
	
	// Display Interactive Controls
	PrintControls();

	size_t currentImage = 0;
	bool liveUpdateWindow = config::initial_live_update_state;
	float learningRate = config::initial_learning_rate;

	// Main Training & UI Loop
	while (true)
	{
		// Pump Windows messages so the viewer window stays responsive
		rendererWindow.ProcessMessages();

		// Handle non-blocking user input
		if (!HandleUserAction(PollUserAction(), network, data, weightsFile, liveUpdateWindow, coordMapper))
		{
			break;
		}

		// Perform batch training step
		float cost;
		if( config::shuffle_pixel_batch )
			cost = RunGradientGuidedTrainingEpoch(network, shuffledInputs, shuffledTargets, currentImage, config::print_every_n_batches, config::batch_size, learningRate, threadPool);
		else
			cost = RunGradientGuidedTrainingEpoch(network, inputSpatialData, targetSpatialData, currentImage, config::print_every_n_batches, config::batch_size, learningRate, threadPool);

		// Live viewer update
		if (liveUpdateWindow)
		{
			rendererWindow.Update(GenerateReconstructedImage(network, data.width * config::output_image_scale, data.height * config::output_image_scale, coordMapper, config::render_mode));
		}

		// Report progress
		std::cout << "COST: " << cost << " LR: " << learningRate << '\n';
	}

	// Final Output Generation & Cleanup
	std::cout << "Generating final output image from network state...\n";

	std::vector<float> finalReconstructedImage = GenerateReconstructedImage(network, data.width * config::output_image_scale, data.height * config::output_image_scale, coordMapper, config::render_mode);

	rendererWindow.Update(finalReconstructedImage);

	saveBMP("network_output.bmp", data.width, data.height, finalReconstructedImage);
	std::cout << "Successfully saved network_output.bmp!\n";

	SaveCheckpoint(network, weightsFile);
	std::cout << "Final " << weightsFile << " saved.\n";
}