#pragma once
#include <functional>
#include <string>
#include <vector>

#include "Types.h"
#include "ImageUtils.h"
#include "Gemini/BMPParser.h"

class Network;
class TrainingThreadPool;

// Appends one row to a training-metrics CSV, writing the header first if the file is new.
void LogTrainingMetrics(const std::string& csvFilepath, size_t batch, engineFloat cost, engineFloat psnr);

// Prints the per-batch training report line, including full-reference metrics when the
// output is at native scale and standard RGB (SSIM needs matching generated/target sizes).
void ReportProgress(engineFloat cost, engineFloat learningRate, const std::vector<engineFloat>& rgbImage,
                    const std::vector<engineFloat>& flatTargetImage, const std::vector<engineFloat>& flatHDImage);

// Logs metrics + saves RGB/gradient frames for the benchmark pipeline. Renders its own
// frames at native resolution, independent of the live viewer / output_image_scale, since
// PSNR/SSIM are only meaningful when compared at the target's native size.
void RunBenchmarkStep(Network& network, const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordMapper,
                      TrainingThreadPool& threadPool, const ImgParser::ImageParsedData& data,
                      const std::vector<engineFloat>& flatTargetImage, size_t currentEpoch, engineFloat cost);

void MakeBenchmarkDirs();
