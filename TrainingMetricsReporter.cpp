#include "TrainingMetricsReporter.h"
#include "Config.h"
#include "Network.h"
#include "ImageGenerator.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

void LogTrainingMetrics(const std::string& csvFilepath, size_t batch, engineFloat cost, engineFloat psnr)
{
	bool writeHeader = !fs::exists(csvFilepath);

	std::ofstream file(csvFilepath, std::ios::app); // Append mode
	if (file.is_open())
	{
		if (writeHeader)
		{
			file << "Batch,Cost,PSNR\n";
		}
		file << batch << "," << cost << "," << psnr << "\n";
	}
}

void ReportProgress(engineFloat cost, engineFloat learningRate, const std::vector<engineFloat>& rgbImage,
                    const std::vector<engineFloat>& flatTargetImage, const std::vector<engineFloat>& flatHDImage)
{
	// TODO: make this more flexible
	// Note: SSIM requires the generated image and target image to be the exact same size.
	// We only calculate this if the scale is 1.0 and we are rendering standard RGB.
	if (config::output_image_scale == 1.0f && config::render_mode == RenderMode::StandardRGB)
	{
		ImageUtils::ImageMetrics metrics = ImageUtils::CalculateFullImageMetrics(rgbImage, flatTargetImage);

		std::cout    << "COST: " << cost 
					<< " LR: " << learningRate 
					<< " | G_SSIM: " << metrics.ssim 
					<< " MAE: " << metrics.mae 
					<< " PSNR(dB): " << metrics.psnr << '\n';
	}
	// Compare out generated image to a ideal HD version (metrics only)
	// A 4x scale means 4x width AND 4x height, so the array is 16x larger hence (output_image_scale * output_image_scale)
	else if ( flatHDImage.size() == ( flatTargetImage.size() * size_t(config::output_image_scale * config::output_image_scale) ) )
	{
		ImageUtils::ImageMetrics metrics = ImageUtils::CalculateFullImageMetrics(rgbImage, flatHDImage);

		std::cout    << "COST: " << cost 
					<< " LR: " << learningRate 
					<< " | G_SSIM: " << metrics.ssim 
					<< " MAE: " << metrics.mae 
					<< " PSNR(dB): " << metrics.psnr << '\n';
	}
	else
	{
		// Report progress
		std::cout << "COST: " << cost << " LR: " << learningRate << " PSNR(dB): " << '\n';
	}
}

void RunBenchmarkStep(Network& network, const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& coordMapper,
                      TrainingThreadPool& threadPool, const BMPParsedData& data,
                      const std::vector<engineFloat>& flatTargetImage, size_t currentEpoch, engineFloat cost)
{
	auto rgbFrame = GenerateReconstructedImage(network, data.width, data.height, coordMapper, RenderMode::StandardRGB, threadPool);
	ImageUtils::ImageMetrics metrics = ImageUtils::CalculateFullImageMetrics(rgbFrame, flatTargetImage);

	// Ensure the output directories exist
	std::string stem = fs::path(config::output_filename).stem().string();
	fs::path basePath(config::output_path);
	auto benchFolder = basePath / ("benchmark_" + stem);
	fs::create_directories(benchFolder);
	fs::create_directories(benchFolder / "rgb");
	fs::create_directories(benchFolder / "grad");

	// Log Metrics to CSV
	std::string csvPath = (benchFolder / "metrics.csv").string();
	LogTrainingMetrics(csvPath, currentEpoch, cost, metrics.psnr);

	// Save Image Frames
	std::string rgbPath = std::format("{}/rgb/step_{:05d}.bmp", benchFolder.string(), currentEpoch);
	saveBMP(rgbPath, data.width, data.height, rgbFrame);

	auto gradFrame = GenerateReconstructedImage(network, data.width, data.height, coordMapper, RenderMode::SpatialGradient, threadPool);
	std::string gradPath = std::format("{}/grad/step_{:05d}.bmp", benchFolder.string(), currentEpoch);
	saveBMP(gradPath, data.width, data.height, gradFrame);
}
