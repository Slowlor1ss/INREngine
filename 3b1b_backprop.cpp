// 3b1b_backprop.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Network.h"

#include <iostream>
#include <chrono>

void Generate(size_t& n1, size_t& n2)
{
	//static size_t callCount = 0;
	//size_t numBits = 2;

	//n1 = callCount % (1u << numBits);
	//n2 = callCount % (1u << numBits);
	//callCount++;

	//size_t numBits = 2;

	//size_t n1, n2;
	//Generate(n1, n2);
	//size_t r = n1 + n2;

	//for (size_t i = 0; i < (numBits + 1); i++)
	//{
	//	input.push_back(float(n1 & 0b1));
	//	n1 = n1 >> 1;
	//}

	//for (size_t i = 0; i < (numBits + 1); i++)
	//{
	//	input.push_back(float(n2 & 0b1));
	//	n2 = n2 >> 1;
	//}


	//for (size_t i = 0; i < (numBits + 1); i++)
	//{
	//	result.push_back(float(r & 0b1));
	//	r = r >> 1;
	//}
}

void GenerateRandom(std::vector<float>& input, std::vector<float>& result)
{
	int x = (rand() % 2);
	int y = (rand() % 2);

	if ((x ^ y))
	{
		result.push_back(1);
		//result.push_back(1);
	}
	else
	{
		result.push_back(0);
		//result.push_back(0);
	}
	input.push_back(x);
	input.push_back(y);
}

#include "Gemini/MNISTReader.h"

void MNISTCheck(const std::vector<std::vector<float>>& images, const std::vector<std::vector<float>>& labels)
{
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
				float value = images[imageidx][y * 28 + x] * 10;
				if (value > 0.1f) c = 'X';
				std::cout << c << " ";
			}
			std::cout << '\n';
		}

		for (size_t i = 0; i < 10; i++)
		{
			std::cout << (labels[imageidx][i]) << ' ';
		}
		std::cout << std::endl;
	}
}

int main()
{	
	auto count = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand((uint32_t)count);

	std::string image_path = "DATA/train-images.idx3-ubyte";
	std::string label_path = "DATA/train-labels.idx1-ubyte";
	std::string test_image_path = "DATA/t10k-images.idx3-ubyte";
	std::string test_label_path = "DATA/t10k-labels.idx1-ubyte";
	std::cout << "Loading MNIST dataset..." << std::endl;

	std::vector<std::vector<float>> images = read_mnist_images(image_path, true);
	std::vector<std::vector<float>> labels = read_mnist_labels(label_path);

	std::vector<std::vector<float>> test_images = read_mnist_images(test_image_path, true);
	std::vector<std::vector<float>> test_labels = read_mnist_labels(test_label_path);

	//MNISTCheck(images, labels);

	std::vector<size_t> layerDims{28*28,64,32,10};
	Network network{ layerDims };


	// train

	float cost = 1.0f;
	size_t batchSize = 32;
	size_t printEveryNBatches = 64;
	float learningRate = 1.0f;

	size_t currentImage = 0;
	while (cost > 0.08f)
	{
		cost = 0.0f;

		for (size_t j = 0; j < printEveryNBatches; j++)
		{
			for (size_t i = 0; i < batchSize; i++)
			{
				float c = network.BackPropagate(images[currentImage], labels[currentImage]);
				cost += c;

				currentImage = (currentImage + 1) % images.size();
			}
			network.ConsumeDelta(learningRate);
			//learningRate = std::max(0.00001f, learningRate * powf(0.99999f, batchSize));
		}


		cost /= (batchSize * printEveryNBatches);
		std::cout << "COST: "
			<< cost
			<< " LR: "
			<< learningRate
			<< '\n';
	}

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

	////network.Serialize();

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