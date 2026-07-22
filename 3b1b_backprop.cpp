// 3b1b_backprop.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Network.h"

#include <iostream>
#include <chrono>

// TODO: don't see the ponit for the database, just use them directly? to have a single instance? does that matter?
// seems to be to use the inheritance more easily >> convert to template instead?
// 
// TODO: Relu needs HE initializtoin: make it so that happens automatically when selecting relu, and not when selecting sigmoid for a layer.
// TODO: serialization
// TOOD: better interface / forntend.

#include "Gemini/MNISTReader.h"
#include "Gemini/CustomFileReader.h"

int main()
{
	auto count = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand((uint32_t)count);

	//std::string image_path = "DATA/fashion/train-images.idx3-ubyte";
	//std::string label_path = "DATA/fashion/train-labels.idx1-ubyte";
	//std::string test_image_path = "DATA/fashion/t10k-images.idx3-ubyte";
	//std::string test_label_path = "DATA/fashion/t10k-labels.idx1-ubyte";
	//std::cout << "Loading MNIST dataset..." << std::endl;

	//std::vector<std::vector<float>> images = read_mnist_images(image_path, true);
	//std::vector<std::vector<float>> labels = read_mnist_labels(label_path);

	//std::vector<std::vector<float>> test_images = read_mnist_images(test_image_path, true);
	//std::vector<std::vector<float>> test_labels = read_mnist_labels(test_label_path);

	ParsedData data;
	ParseBinaryData("C:\\Github\\TRP\\TempleRunPreprocess\\myoutput.bin", data);

	std::vector<std::vector<float>> images = data.inputs;
	std::vector<std::vector<float>> labels = data.outputs;

	std::vector<std::vector<float>> test_images = images;
	std::vector<std::vector<float>> test_labels = labels;

	std::vector<size_t> layerDims{ 12,24,9 };
	Network network{ layerDims };


	// trainzl

	float cost = 1.0f;
	size_t batchSize = 4096 * 4;
	size_t printEveryNBatches = 16;
	float learningRate = 10.0f; // usually would be alot lower

	size_t currentImage = 0;

	int writebackCtr = 0;
	while (cost > 0.01f)
	{
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
			learningRate = learningRate * pow(0.99999999, batchSize);
		}

		// batchSize = (batchSize + 1) % 128;

		cost /= (batchSize * printEveryNBatches);
		std::cout << "COST: "
			<< cost
			<< " LR: "
			<< learningRate
			<< '\n';

		writebackCtr++;
		if (writebackCtr % 20 == 0)
		{
			std::ofstream file{ "weights_biases.csv" };
			network.Serialize(file);
			std::cout << "writeback completed\n";
		}
	}

#if 0
	// evaluate:
	float correct = 0;
	float resultCost = 0.0f;
	for (size_t i = 0; i < test_images.size(); i++)
	{
		auto result = network.Propagate(test_images[i]);
		float c = network.BackPropagate(test_images[i], test_labels[i]);


		size_t maxI = 0;
		for (size_t i = 1; i < 10; i++)
		{
			if (result[i] >= result[maxI])
			{
				maxI = i;
			}
		}
		if (test_labels[i][maxI] == 1)
		{
			correct += 1.0f;
		}


		resultCost += c;
	}
	correct /= test_images.size();
	resultCost /= test_images.size();
	std::cout << "RESULT COST: "
		<< resultCost
		<< " ACCURACY: "
		<< correct
		<< '\n';

	// print first image in console:
	while (true) {
		int imageidx;
		std::cin >> imageidx;
		imageidx %= images.size();

		for (size_t y = 0; y < 28; y++)
		{
			for (size_t x = 0; x < 28; x++)
			{
				char c = ' ';
				float value = test_images[imageidx][y * 28 + x] * 10;
				if (value > 0.1f) c = 'X';
				std::cout << c << " ";
			}
			std::cout << '\n';
		}
		auto result = network.Propagate(test_images[imageidx]);


		for (size_t i = 0; i < 10; i++)
		{
			std::cout << i << '\t' << (test_labels[imageidx][i]) << '\t' << result[i] << '\n';
		}
		std::cout << std::endl;

	}
#endif

	std::ofstream file{"weights_biases.csv"};
	network.Serialize(file);

	//while (true)
	//{
	//	std::vector<float> inputs;
	//	inputs.resize(layerDims[0]);
	//	for (size_t i = 0; i < inputs.size(); i++)
	//	{
	//		std::cin >> inputs[i];
	//	}

	//	auto result = network.Propagate(inputs);
	//	
	//	for (float activation : result)
	//	{
	//		std::cout << activation << " ";
	//	}
	//	std::cout << std::endl;
	//}
}