void NeuralImageRecreator()
{
	// Load Input Data & Derive Checkpoint Filename
	const std::string weightsFile = GetCheckpointFilename(config::output_filename, config::output_path);
	// Alternative image loading: ParseBMPData("Sarah_large.bmp", data);

	BMPParsedData data;
	ParseBMPData(config::target_image_file.c_str(), data);

	// Setup our coordinate mapper lambda for the ImageGenerator
	std::function<MappedInput(float, float)> coordMapper = nullptr;
	if (config::use_positional_encoding) {
#if _HAS_CXX23
		// Don't use bind as appenrently it generates horrible assembly and is hard for the compiler to optimize
		// Actually c++ 23 allows us to use bind back this is apperently eveything std::bind always wanted to be :))!
		coordMapper = std::bind_back(PositionalEncodeWithDerivatives, config::pe_num_frequencies);
#else
		// Reject modern C++ return to Monke 
		coordMapper = [](float x, float y) {
			return PositionalEncodeWithDerivatives(x, y, config::pe_num_frequencies);
		};
#endif
	}

	// Prepare images dataset based on the current mode
	std::vector<std::vector<float>> images;
	if (config::use_positional_encoding) {
		images.reserve(data.inputs.size());
		for (const auto& rawInput : data.inputs) {
			images.push_back(PositionalEncode(rawInput[0], rawInput[1], config::pe_num_frequencies));
		}
	} else {
		images = data.inputs;
	}

	const std::vector<std::vector<float>>& labels = data.outputs;

	// Initialize Neural Network & Visualizer Window
	size_t inputLayerSize = config::use_positional_encoding ? (config::pe_num_frequencies * 4) : 2;
	std::vector<size_t> layerDims{ inputLayerSize, 8, 16, 32, 64, 64, 32, 16, 3 };
	
	// Pass the activation functions dynamically!
	Network network{ 
		layerDims, 
		ActFunc::DataBase::FindActFunc<ActFunc::Siren>(),
		// TODO: look in to this more maybe just use sigmoid as its basically the same or none as its more truthfully ig
		// and the docmentation says to just use a linear or sine https://deepwiki.com/vsitzmann/siren/2-siren-architecture#sinelayer-and-network-structure
		ActFunc::DataBase::FindActFunc<ActFunc::ColorSquash>() 
	};

	ImageWindow rendererWindow(data.width, data.height);

	// Load Weights Checkpoint (Validates dimensions vs current layerDims automatically)
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
		float cost = RunTrainingEpoch(network, images, labels, currentImage, config::print_every_n_batches, config::batch_size, learningRate);

		// Live viewer update
		if (liveUpdateWindow)
		{
			rendererWindow.Update(GenerateReconstructedImage(network, data.width, data.height, coordMapper, config::render_mode));
		}

		// Report progress
		std::cout << "COST: " << cost << " LR: " << learningRate << '\n';
	}

	// Final Output Generation & Cleanup
	std::cout << "Generating final output image from network state...\n";

	std::vector<float> finalReconstructedImage = GenerateReconstructedImage(network, data.width, data.height, coordMapper, config::render_mode);

	rendererWindow.Update(finalReconstructedImage);

	saveBMP("network_output.bmp", data.width, data.height, finalReconstructedImage);
	std::cout << "Successfully saved network_output.bmp!\n";

	SaveCheckpoint(network, weightsFile);
	std::cout << "Final " << weightsFile << " saved.\n";
}