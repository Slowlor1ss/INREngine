#pragma once
#include <functional>
#include <vector>

#include "Types.h"
#include "ImageUtils.h"
#include "Gemini/BMPParser.h"

// Encoded inputs/targets ready for training, plus whichever raw pixel coordinates line up
// with them (identity order, or shuffled order -- see BuildSpatialDataset).
struct SpatialDataset
{
	std::vector<ImageUtils::SpatialData> inputs;
	std::vector<ImageUtils::SpatialData> targets;
	std::vector<engineFloat> pixelX;
	std::vector<engineFloat> pixelY;
};

// Builds the coordinate -> SpatialData mapping lambda used everywhere a raw (x, y) needs to
// become network input (standard PE, Gaussian PE, or raw xy passthrough), based on config.
std::function<ImageUtils::SpatialData(engineFloat, engineFloat)> BuildCoordMapper();

// Generates encoded inputs (coords/PE + spatial slopes) and targets (RGB + spatial edges)
// from the parsed image, optionally shuffled per config::shuffle_pixel_batch. Only allocates
// the shuffled copies when actually shuffling.
SpatialDataset BuildSpatialDataset(const BMPParsedData& data,
                                   const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordMapper);

// Flattens the target image into a single RGB buffer for full-reference metrics (SSIM/PSNR/etc).
std::vector<engineFloat> FlattenTargetImage(const BMPParsedData& data);
