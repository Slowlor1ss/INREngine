#include "TrainingUserActions.h"
#include "Config.h"
#include "CheckpointUtils.h"
#include "CudaManager.cuh"
#include "ImageGenerator.h"

#include <conio.h> // For _kbhit() and _getch()
#include <iostream>
#include <vector>
#include "Gemini/ImageParser.h"

void PrintControls()
{
	std::cout << "Training started.\n"
		<< " [Q/ESC] - Stop training\n"
		<< " [W]     - Save weights\n"
		<< " [D]     - Swap render mode (Used for debugging)\n"
		<< " [E]     - Save image to disk\n"
		<< " [V]     - Toggle live viewer window\n\n";
}

// Polls non-blocking keyboard input buffer and maps to UserAction.
UserAction PollUserAction()
{
	if (!_kbhit()) return UserAction::None;

	switch (int ch = _getch())
	{
		case 'q': case 'Q': case 27: // ESC
			return UserAction::Quit;
		case 'w': case 'W':
			return UserAction::SaveWeights;
		case 'e': case 'E':
			return UserAction::ExportImage;
		case 'v': case 'V':
			return UserAction::ToggleViewer;
		case 'd': case 'D':
			return UserAction::SwapRenderMode;
		default:
			return UserAction::None;
	}
}

// Renders the network at the configured output scale and saves it as network_output.bmp.
// Split out of HandleUserAction() since it's a fairly large, self-contained chunk of work.
static void ExportNetworkImage(Network& network, GpuNetwork* gpuNet, const ImgParser::ImageParsedData& data,
                               const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& mapper,
                               TrainingThreadPool& threadPool)
{
	int renderWidth = int(data.width * config::output_image_scale);
	int renderHeight = int(data.height * config::output_image_scale);
	int totalRenderPixels = renderWidth * renderHeight;
	std::vector<engineFloat> reconstructedImage;

	if (config::use_gpu)
	{
		size_t outChannels = network.GetLayers().back()->GetNumNeurons();

		engineFloat* d_exportPixelX = nullptr;
		engineFloat* d_exportPixelY = nullptr;
		engineFloat* d_exportInputs = nullptr;
		engineFloat* d_exportColors = nullptr;

		CUDA_CHECK(cudaMalloc(&d_exportColors, totalRenderPixels * outChannels * sizeof(engineFloat)));

		if (config::use_grid_encoding)
		{
			CUDA_CHECK(cudaMalloc(&d_exportPixelX, totalRenderPixels * sizeof(engineFloat)));
			CUDA_CHECK(cudaMalloc(&d_exportPixelY, totalRenderPixels * sizeof(engineFloat)));

			std::vector<engineFloat> h_exportPixelX(totalRenderPixels);
			std::vector<engineFloat> h_exportPixelY(totalRenderPixels);

			for (int y = 0; y < renderHeight; ++y)
			{
				for (int x = 0; x < renderWidth; ++x)
				{
					int idx = y * renderWidth + x;
					// Match the [-1.0, 1.0] domain so FindCell maps 0.0 to 1.0 across the full grid
					h_exportPixelX[idx] = (static_cast<engineFloat>(x) / static_cast<engineFloat>(renderWidth)) * 2.0f - 1.0f;
					h_exportPixelY[idx] = (static_cast<engineFloat>(y) / static_cast<engineFloat>(renderHeight)) * 2.0f - 1.0f;
				}
			}

			CUDA_CHECK(cudaMemcpy(d_exportPixelX, h_exportPixelX.data(), totalRenderPixels * sizeof(engineFloat), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_exportPixelY, h_exportPixelY.data(), totalRenderPixels * sizeof(engineFloat), cudaMemcpyHostToDevice));
		}
		else
		{
			size_t inChannels = network.GetLayers().front()->GetNumNeurons();
			CUDA_CHECK(cudaMalloc(&d_exportInputs, totalRenderPixels * inChannels * sizeof(engineFloat)));
			std::vector<engineFloat> h_exportInputs(totalRenderPixels * inChannels);

			for (int y = 0; y < renderHeight; ++y)
			{
				for (int x = 0; x < renderWidth; ++x)
				{
					int pixelIdx = (y * renderWidth + x) * inChannels;
					engineFloat normX = (static_cast<engineFloat>(x) / static_cast<engineFloat>(renderWidth)) * 2.0f - 1.0f;
					engineFloat normY = (static_cast<engineFloat>(y) / static_cast<engineFloat>(renderHeight)) * 2.0f - 1.0f;

					ImageUtils::SpatialData encoded = mapper(normX, normY);
					for (size_t c = 0; c < encoded.values.size(); ++c) {
						h_exportInputs[pixelIdx + c] = encoded.values[c];
					}
				}
			}
			CUDA_CHECK(cudaMemcpy(d_exportInputs, h_exportInputs.data(), h_exportInputs.size() * sizeof(engineFloat), cudaMemcpyHostToDevice));
		}

		gpuNet->PredictGPU(d_exportPixelX, d_exportPixelY, d_exportInputs, d_exportColors, totalRenderPixels);

		std::vector<engineFloat> h_colors(totalRenderPixels * outChannels);
		CUDA_CHECK(cudaMemcpy(h_colors.data(), d_exportColors, h_colors.size() * sizeof(engineFloat), cudaMemcpyDeviceToHost));

		if (outChannels == 3)
		{
			reconstructedImage = h_colors;
		}
		else if (outChannels == 1)
		{
			reconstructedImage.resize(size_t(totalRenderPixels) * 3);
			for (int i = 0; i < totalRenderPixels; ++i) {
				reconstructedImage[i * 3 + 0] = h_colors[i];
				reconstructedImage[i * 3 + 1] = h_colors[i];
				reconstructedImage[i * 3 + 2] = h_colors[i];
			}
		}

		if (d_exportPixelX) cudaFree(d_exportPixelX);
		if (d_exportPixelY) cudaFree(d_exportPixelY);
		if (d_exportInputs) cudaFree(d_exportInputs);
		if (d_exportColors) cudaFree(d_exportColors);
	}
	else
	{
		reconstructedImage = GenerateReconstructedImage(network, renderWidth, renderHeight, mapper, config::render_mode, threadPool);
	}

	ImageParser::Save("network_output.bmp", renderWidth, renderHeight, reconstructedImage);
	std::cout << "Successfully saved network_output.bmp!\n";
}

// Handles user actions outside the main training loop; returns false to break loop.
bool HandleUserAction(const UserAction action, Network& network, GpuNetwork* gpuNet, const ImgParser::ImageParsedData& data,
                       const std::string& weightsFile, bool& liveUpdateWindow,
                       const std::function<ImageUtils::SpatialData(engineFloat, engineFloat)>& mapper,
                       TrainingThreadPool& threadPool)
{
	switch (action)
	{
		case UserAction::Quit:
			std::cout << "\n[Interrupt Received] Stopping training...\n";
			return false;

		case UserAction::SaveWeights:
			SaveCheckpoint(network, gpuNet, weightsFile);
			break;

		case UserAction::ExportImage:
			ExportNetworkImage(network, gpuNet, data, mapper, threadPool);
			break;

		case UserAction::ToggleViewer:
			liveUpdateWindow = !liveUpdateWindow;
			std::cout << (liveUpdateWindow ? "Live viewer window ENABLED\n" : "Live viewer window DISABLED\n");
			break;

		case UserAction::SwapRenderMode:
			config::render_mode = (RenderMode)(((int)config::render_mode + 1) % (int)RenderMode::Last);
			std::cout << "Updated render mode!\n"; // Im not making an enum to sting >:(
			break;

		case UserAction::None:
			break;
	}

	return true;
}
