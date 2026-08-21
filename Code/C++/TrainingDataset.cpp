#include "TrainingDataset.h"
#include "Config.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

std::function<ImageUtils::SpatialData(engineFloat, engineFloat)> BuildCoordMapper()
{
	std::function<ImageUtils::SpatialData(engineFloat, engineFloat)> coordMapper = nullptr;

	if (config::use_positional_encoding)
	{
		coordMapper = [freqs = config::pe_num_frequencies](engineFloat x, engineFloat y) {
			// No longer scaling by 2.0/w! Keep it pure.
			return ImageUtils::PositionalEncodeWithDerivatives(x, y, freqs);
		};
	}
	else if (config::use_gaussian_pe)
	{
		ImageUtils::GaussianPositionalEncoder gaussianPE(64, 10.0f);
		// TODO: maybe find a better way but
		// We *copy* gaussianPE into the lambda so it lives forever so dont pass as reference or BOOM
		coordMapper = [gaussianPE](engineFloat x, engineFloat y) {
			return gaussianPE(x, y);
		};
	}
	else
	{
		coordMapper = [](engineFloat x, engineFloat y) {
			ImageUtils::SpatialData d;
			d.values = { x, y };
			d.gradX = { 1.0f, 0.0f }; // Pure normalized derivative
			d.gradY = { 0.0f, 1.0f }; // Pure normalized derivative
			return d;
		};
	}

	return coordMapper;
}

SpatialDataset BuildSpatialDataset(const ImgParser::ImageParsedData& data,
                                   const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordMapper)
{
	SpatialDataset dataset;

	// Generate Target Outputs (RGB + Spatial Edges)
	dataset.targets = ImageUtils::GenerateGradientTargets(data.outputs, data.width, data.height);

	// Generate Inputs (Coordinates/PE + Spatial Slopes)
	dataset.inputs.reserve(data.inputs.size());
	for (const auto& rawInput : data.inputs) {
		dataset.inputs.push_back(coordMapper(rawInput[0], rawInput[1]));
	}

	if (config::shuffle_pixel_batch)
	{
		std::cout << "Shuffling dataset for random pixel batching...\n";

		// Create an array of indices [0, 1, 2, ... 65535]
		std::vector<size_t> indices(dataset.inputs.size());
		std::iota(indices.begin(), indices.end(), 0);

		// Shuffle the indices
		std::mt19937 g(42); // Fixed seed for consistency
		std::shuffle(indices.begin(), indices.end(), g);

		std::vector<ImageUtils::SpatialData> shuffledInputs(dataset.inputs.size());
		std::vector<ImageUtils::SpatialData> shuffledTargets(dataset.targets.size());
		dataset.pixelX.resize(dataset.inputs.size());
		dataset.pixelY.resize(dataset.inputs.size());

		for (size_t i = 0; i < indices.size(); ++i) {
			shuffledInputs[i] = dataset.inputs[indices[i]];
			shuffledTargets[i] = dataset.targets[indices[i]];
			// Map the raw coordinates alongside the shuffled targets!
			dataset.pixelX[i] = data.inputs[indices[i]][0];
			dataset.pixelY[i] = data.inputs[indices[i]][1];
		}

		dataset.inputs = std::move(shuffledInputs);
		dataset.targets = std::move(shuffledTargets);
	}
	else
	{
		dataset.pixelX.resize(dataset.inputs.size());
		dataset.pixelY.resize(dataset.inputs.size());
		for (size_t i = 0; i < dataset.inputs.size(); ++i) {
			dataset.pixelX[i] = data.inputs[i][0];
			dataset.pixelY[i] = data.inputs[i][1];
		}
	}

	return dataset;
}

std::vector<engineFloat> FlattenTargetImage(const ImgParser::ImageParsedData& data)
{
	std::vector<engineFloat> flatTargetImage;
	if (!data.outputs.empty())
	{
		flatTargetImage.reserve(data.outputs.size() * data.outputs[0].size());
		for (const auto& pixel : data.outputs) {
			flatTargetImage.insert(flatTargetImage.end(), pixel.begin(), pixel.end());
		}
	}
	return flatTargetImage;
}
