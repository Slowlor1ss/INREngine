// 3b1b_backprop.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Network.h"

#include <iostream>
#include <fstream>
#include <chrono>

#include <windows.h>

// TODO: don't see the ponit for the database, just use them directly? to have a single instance? does that matter?
// seems to be to use the inheritance more easily >> convert to template instead?
// 
// TODO: Relu needs HE initializtoin: make it so that happens automatically when selecting relu, and not when selecting sigmoid for a layer.
// TODO: serialization
// TOOD: better interface / forntend.

#include "Gemini/MNISTReader.h"
#include "Gemini/BMPParser.h"
#include "Gemini/CustomFileReader.h"
#include "WindowRenderer.h" // <-- Included our new renderer

#include <conio.h> // For _kbhit() and _getch()

int main()
{
	auto count = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand((uint32_t)count);

	BMPParsedData data;

	ParseBMPData("Sarah_large.bmp", data);

	std::vector<std::vector<float>> images = data.inputs;
	std::vector<std::vector<float>> labels = data.outputs;

	std::vector<std::vector<float>> test_images = images;
	std::vector<std::vector<float>> test_labels = labels;

	std::vector<size_t> layerDims{ 2,32,32,32,32,32,32,32,32,32,32,32,3 };
	Network network{ layerDims };

	// Create our visualizer window
	ImageWindow rendererWindow(data.width, data.height);

	// Try loading pre-existing checkpoint if available
	{
		std::ifstream inFile{ "weights_biases.csv" };
		if (inFile.is_open())
		{
			network.Deserialize(inFile);
			std::cout << "Loaded existing weights and biases checkpoint.\n";
		}
	}

	std::cout << "Training started.\n";
	std::cout << " [Q/ESC] - Stop training\n";
	std::cout << " [W]     - Save weights\n";
	std::cout << " [E]     - Save image to disk\n";
	std::cout << " [V]     - Update live viewer window\n";

	float cost = 1.0f;
	size_t batchSize = 4096;
	size_t printEveryNBatches = 1;
	float learningRate = 1.0f; // usually would be alot lower

	size_t currentImage = 0;

	int writebackCtr = 0;
	bool liveUpdateWindow = true;
	while (true)
	{
		// Pump Windows messages so our renderer window doesn't freeze
		rendererWindow.ProcessMessages();

		// --- THE NEW INTERRUPT CHECK ---
		// _kbhit() returns true instantly if a key is waiting in the buffer
		if (_kbhit())
		{
			int ch = _getch(); // Read the key
			if (ch == 'q' || ch == 'Q' || ch == 27) // 27 is the ASCII code for ESC
			{
				std::cout << "\n[Interrupt Received] Stopping training...\n";
				break; // Exit the while loop
			}

			if (ch == 'w' || ch == 'W')
			{
				std::ofstream file{ "weights_biases.csv" };
				network.Serialize(file);
				std::cout << "Weights saved to disk!" << std::endl;
			}

			if (ch == 'e' || ch == 'E' || ch == 'v' || ch == 'V') 
			{
				// Allocate space for the image we are going to write
				std::vector<float> reconstructed_image(data.width * data.height * 3);

				// Loop through every (X, Y) coordinate and ask the network what color it thinks it is
				for (int y = 0; y < data.height; ++y) {
					for (int x = 0; x < data.width; ++x) {

						// Normalize X and Y exactly like in training
						std::vector<float> input = { x / float(data.width), y / float(data.height) };
						std::vector<float> output = network.Propagate(input);

						int pixel_index = (y * data.width + x) * 3;
						reconstructed_image[pixel_index + 0] = output[0]; // R
						reconstructed_image[pixel_index + 1] = output[1]; // G
						reconstructed_image[pixel_index + 2] = output[2]; // B
					}
				}

				if (ch == 'e' || ch == 'E') 
				{
					saveBMP("network_output.bmp", data.width, data.height, reconstructed_image);
					std::cout << "Successfully saved network_output.bmp!" << std::endl;
				} 
				else if (ch == 'v' || ch == 'V') 
				{
					liveUpdateWindow = true;
					//rendererWindow.Update(reconstructed_image);
					liveUpdateWindow ? 
					std::cout << "Start live viewer window!" << '\n'
					:
					std::cout << "Stop live viewer window!" << '\n';
				}
			}
			
		}
		// -------------------------------
		
		cost = 0.0f;

		for (size_t j = 0; j < printEveryNBatches; j++)
		{
			for (size_t i = 0; i < batchSize; i++)
			{
				float c = network.BackPropagate(images[currentImage], labels[currentImage]);
				cost += c;

				currentImage = rand() % images.size(); // (currentImage + 1) % images.size();
			}

			network.ConsumeDelta(learningRate);
			learningRate = learningRate * pow(0.9999999, batchSize);
		}
		
		if (liveUpdateWindow)
		{
			// Allocate space for the image we are going to write
			std::vector<float> reconstructed_image(data.width * data.height * 3);

			// Loop through every (X, Y) coordinate and ask the network what color it thinks it is
			for (int y = 0; y < data.height; ++y) {
				for (int x = 0; x < data.width; ++x) {

					// Normalize X and Y exactly like in training
					std::vector<float> input = { x / float(data.width), y / float(data.height) };
					std::vector<float> output = network.Propagate(input);

					int pixel_index = (y * data.width + x) * 3;
					reconstructed_image[pixel_index + 0] = output[0]; // R
					reconstructed_image[pixel_index + 1] = output[1]; // G
					reconstructed_image[pixel_index + 2] = output[2]; // B
				}
			}
			
			rendererWindow.Update(reconstructed_image);
		}

		cost /= (batchSize * printEveryNBatches);
		std::cout << "COST: "
			<< cost
			<< " LR: "
			<< learningRate
			<< '\n';
	}

	std::cout << "Generating output image from network state...\n";

	// Allocate space for the image we are going to write
	std::vector<float> reconstructed_image(data.width * data.height * 3);

	// Loop through every (X, Y) coordinate and ask the network what color it thinks it is
	for (int y = 0; y < data.height; ++y) {
		for (int x = 0; x < data.width; ++x) {

			// Normalize X and Y exactly like in training
			std::vector<float> input = { x / float(data.width), y / float(data.height) };
			std::vector<float> output = network.Propagate(input);

			int pixel_index = (y * data.width + x) * 3;
			reconstructed_image[pixel_index + 0] = output[0]; // R
			reconstructed_image[pixel_index + 1] = output[1]; // G
			reconstructed_image[pixel_index + 2] = output[2]; // B
		}
	}

	// Update the viewer one last time before saving
	rendererWindow.Update(reconstructed_image);

	// Save the image
	saveBMP("network_output.bmp", data.width, data.height, reconstructed_image);
	std::cout << "Successfully saved network_output.bmp!" << std::endl;

	// Final network serialization
	std::ofstream file{ "weights_biases.csv" };
	network.Serialize(file);
	std::cout << "Final weights_biases.csv saved." << std::endl;
}