static float RunGradientGuidedTrainingEpoch(Network& network,
const std::vector<ImageUtils::SpatialData>& inputs,
const std::vector<ImageUtils::SpatialData>& targets,
size_t& currentImageIdx,
const size_t printEveryNBatches,
const size_t batchSize,
float& learningRate)
{
	float totalCost = 0.0f;
	Network::SpatialDerivativeBuffer spatialBuffers; // Create one buffer to reuse

	for (size_t j = 0; j < printEveryNBatches; j++)
	{
		for (size_t i = 0; i < batchSize; i++)
		{
			const auto& input = inputs[currentImageIdx];
			const auto& target = targets[currentImageIdx];

			// 1. FORWARD PASS
			network.PropagateSpatialDerivativesThreadSafe(input.values, input.gradX, input.gradY, spatialBuffers);

			// 2. BACKWARD PASS (Applies weight updates instantly)
			network.BackPropagateGradientGuided(target, input.values, spatialBuffers.activations, spatialBuffers.preActivations, spatialBuffers, learningRate);

			// Calculate standard RGB cost for the console log
			float cost = 0.0f;
			for(size_t c = 0; c < target.values.size(); ++c){
				float diff = spatialBuffers.activations.back()[c] - target.values[c]; // TODO: all activations become nan
				cost += diff * diff;
				if ( std::_Is_nan(cost) )
				{
					//std::cout << diff;
					//__debugbreak();
					//cost = 0.f;
				}
			}
			totalCost += cost / float(target.values.size());
			if ( std::_Is_nan(totalCost) )
			{
				totalCost = 999'999.f;
				//std::cout << cost;
				//__debugbreak();
			}
			//totalCost = std::clamp( totalCost, 0.f, 999999.f);

			currentImageIdx = GetRandomImageIndex(inputs.size());
		}
		
		// Apply the averaged batch weights!
		network.ConsumeDelta(learningRate);
        
		// Decay learning rate
		learningRate *= static_cast<float>(std::pow(0.9999999, batchSize));
	}

	return totalCost / static_cast<float>(batchSize * printEveryNBatches);
}

void NeuralImageRecreator()
{
	// Load Input Data & Derive Checkpoint Filename
	const std::string weightsFile = GetCheckpointFilename(config::output_filename, config::output_path);

	BMPParsedData data;
	ParseBMPData(config::target_image_file.c_str(), data);

	// Setup our coordinate mapper lambda for the ImageGenerator
	std::function<ImageUtils::SpatialData(float, float)> coordMapper = nullptr;
if (config::use_positional_encoding) {
		coordMapper = [w = static_cast<float>(data.width), h = static_cast<float>(data.height), freqs = config::pe_num_frequencies](float x, float y) {
			ImageUtils::SpatialData d = PositionalEncodeWithDerivatives(x, y, freqs);
			// NATIVELY SCALE TO PIXEL SPACE!
			// Apply the chain rule: multiply all X derivatives by (1.0 / width) 
			// and all Y derivatives by (1.0 / height)
			for (float& gx : d.gradX) {
				gx /= w;
			}
			for (float& gy : d.gradY) {
				gy /= h;
			}
			
			return d;
		};
	} 
	else {
		// Fallback mapper for standard inputs
		coordMapper = [w = static_cast<float>(data.width), h = static_cast<float>(data.height)](float x, float y) {
				ImageUtils::SpatialData d;
				d.values = { x, y };
				// NATIVELY SCALE TO PIXEL SPACE! 
				// Instead of 1.0, X changes by 1 pixel width.
				d.gradX = { 1.0f / w, 0.0f };
				d.gradY = { 0.0f, 1.0f / h };
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

	// Initialize Neural Network & Visualizer Window
	size_t inputLayerSize = config::use_positional_encoding ? (config::pe_num_frequencies * 4) : 2;
	//std::vector<size_t> layerDims{ inputLayerSize, 8, 16, 32, 64, 64, 32, 16, 3 };
	//std::vector<size_t> layerDims{ inputLayerSize, 64, 64, 64, 64, 3 };
	//std::vector<size_t> layerDims{ inputLayerSize, 128, 128, 3 };
	//std::vector<size_t> layerDims{ inputLayerSize, 8, 16, 32, 16, 8, 3 };
	std::vector<size_t> layerDims;
	if (!config::custom_layer_dims.empty()) {
		layerDims.insert(layerDims.end(), config::custom_layer_dims.begin(), config::custom_layer_dims.end());
	} else {
		layerDims.insert(layerDims.end(), { 64, 64, 64, 64, 3 });
	}
	
	// Pass the activation functions dynamically!
	Network network{ 
		layerDims, 
		ActFunc::DataBase::FindActFunc<ActFunc::Siren>(),
		// TODO: look in to this more maybe just use sigmoid as its basically the same or none as its more truthfully ig
		// and the docmentation says to just use a linear or sine https://deepwiki.com/vsitzmann/siren/2-siren-architecture#sinelayer-and-network-structure
		ActFunc::DataBase::FindActFunc<ActFunc::None>() 
	};

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
		float cost = RunGradientGuidedTrainingEpoch(network, inputSpatialData, targetSpatialData, currentImage, config::print_every_n_batches, config::batch_size, learningRate);
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