#pragma once

#include "helpers.h"

inline void RunMNISTAutoencoder()
{
	// Set to true to explore the Latent Space!
	bool isTestMode = false;
	if (isTestMode)
	{
		std::cout << "==================================\n\n";

		std::vector<Network::LayerInfo> layerDims{
			{2, ActFunc::DataBase::FindActFunc<ActFunc::Empty>()},
			{16, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{128, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{256, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{28 * 28, ActFunc::DataBase::FindActFunc<ActFunc::Sigmoid>()} };

		Network network{ layerDims, CostFunc::DataBase::FindCostFunc<CostFunc::L1>() };

		LoadCheckpoint(network, "mnist_decoder.csv");

		int scale = 10;
		ImageWindow rendererWindow(28 * scale, 28 * scale);

		float latent_x = 0.0f;
		float latent_y = 0.0f;
		float step = 1.0f; // How fast you move through the space

		std::cout << "Controls:\n [W/A/S/D] - Move in Latent Space\n [Q/ESC]   - Quit\n\n";
		std::cout << "Current Coordinate: (" << latent_x << ", " << latent_y << ")\r";

		while (true)
		{
			rendererWindow.ProcessMessages();

			bool changed = true; // Force render on first frame
			if (_kbhit())
			{
				int ch = _getch();
				switch (ch)
				{
				case 'w': case 'W': latent_y += step; changed = true; break;
				case 's': case 'S': latent_y -= step; changed = true; break;
				case 'a': case 'A': latent_x -= step; changed = true; break;
				case 'd': case 'D': latent_x += step; changed = true; break;
				case 'q': case 'Q': case 27: return;
				}
			}

			if (changed)
			{
				// The \r and trailing spaces keep the console clean on a single line
				std::cout << "Current Coordinate: (" << latent_x << ", " << latent_y << ")          \r";

				// Run the forward pass and update the screen
				rendererWindow.Update(GenerateImageFromLatent(network, latent_x, latent_y , scale));
			}

			// Small sleep to prevent frying your CPU (roughly 60 FPS)
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}
	}
	else
	{
		// 2. Load Input Data & Derive Checkpoint Filename
		const std::string weightsFile = "mnist_wb/encode_decode.csv";

		std::vector<std::vector<float>> images, labels, testImages, testLabels;

		LoadMNISTData(images, labels, testImages, testLabels);

		labels = images;
		testLabels = testImages;

		// 3. Initialize Neural Network & Visualizer Window
		std::vector<Network::LayerInfo> layerDims{
			{28 * 28, ActFunc::DataBase::FindActFunc<ActFunc::Empty>()},
			{256, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{128, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{2, ActFunc::DataBase::FindActFunc<ActFunc::None>()},
			{128, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{256, ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>()},
			{28 * 28, ActFunc::DataBase::FindActFunc<ActFunc::Sigmoid>()}
		};
		Network network{ layerDims, CostFunc::DataBase::FindCostFunc<CostFunc::L1>() };

		ImageWindow rendererWindow(28 * 2, 28);

		// 4. Load Weights Checkpoint (Validates dimensions vs current layerDims automatically)
		LoadCheckpoint(network, weightsFile);

		// 5. Display Interactive Controls
		PrintControls();

		// 6. Hyperparameters & Training State
		const size_t batchSize = 32;
		const size_t printEveryNBatches = 5;
		float learningRate = 0.001f;

		size_t currentImage = 0;
		bool liveUpdateWindow = true;

		// 7. Main Training & UI Loop
		while (true)
		{
			// Pump Windows messages so the viewer window stays responsive
			rendererWindow.ProcessMessages();

			// Handle non-blocking user input
			if (!HandleUserAction(PollUserAction(), network, weightsFile, liveUpdateWindow))
			{
				break;
			}

			// Perform batch training step
			float cost = RunTrainingEpoch(network, images, labels, currentImage, printEveryNBatches, batchSize, learningRate);

			// Live viewer update
			if (liveUpdateWindow)
			{
				rendererWindow.Update(GenerateReconstructedImageMNIST(network, images[GetRandomImageIndex(images.size())]));
			}

			// Report progress
			std::cout << "COST: " << cost << " LR: " << learningRate << '\n';
		}

		// 8. Final Output Generation & Cleanup
		std::cout << "Generating final output image from network state...\n";

		SaveCheckpoint(network, weightsFile);
		std::cout << "Final " << weightsFile << " saved.\n";
	}
}
