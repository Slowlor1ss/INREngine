#include "Parameters.h"
#include <sstream>
#include <cassert>
#include <cmath> // For sqrtf


Parameters::Parameters(size_t numBiases, size_t numWeights, ActFunc::Base* actFunc, size_t lIdx)
{
	biases.resize(numBiases, 0.0f);
	weights.resize(numWeights, 0.0f);

	for (size_t i = 0; i < numBiases; i++)
	{
		biases[i] = 0.0f; // Biases are safer to initialize as straight 0
	}

	static std::random_device rd;
	static std::mt19937 generator(rd());
	size_t fanIn = (numBiases > 0) ? (numWeights / numBiases) : 1;
	size_t fanOut = numBiases;// wrong!!!! should be number of neurons in the next layer >> outoging connections per neuron
	for (size_t i = 0; i < numWeights; i++)
	{
		weights[i] = actFunc->GenerateInitialWeight( generator, fanIn, fanOut );
	}

	layerIdx = lIdx;
}

Parameters& Parameters::operator+=(const Parameters& other)
{
	if (biases.size() == other.biases.size())
	{
		for (size_t j = 0; j < biases.size(); j++)
		{
			biases[j] += other.biases[j];
		}
	}

	if (weights.size() == other.weights.size())
	{
		for (size_t j = 0; j < weights.size(); j++)
		{
			weights[j] += other.weights[j];
		}
	}

	return *this;
}

Parameters& Parameters::operator*=(float other)
{
	for (size_t j = 0; j < biases.size(); j++)
	{
		biases[j] *= other;
	}
	for (size_t j = 0; j < weights.size(); j++)
	{
		weights[j] *= other;
	}

	return *this;
}

void Parameters::Clear()
{
	(*this) *= 0.0f;
}

// Inherited via Serializable
void Parameters::Deserialize(std::istream& in) {};

//
void Parameters::Serialize(std::ostream& out)
{
	char delim[3] = ", ";

	out << "static const uint num_biases_l" << layerIdx << " = " << biases.size() << ";";

	if (layerIdx != 0)
	{
		out << "float biases_l" << layerIdx << "[num_biases_l" << layerIdx << "] = {";
		for (size_t i = 0; i < biases.size(); i++)
		{
			out << biases[i] << delim;
		}
		out << "};\n";
	}

	out << "static const uint num_weights_l" << layerIdx << " = " << weights.size() << ";";
	if (layerIdx != 0)
	{
		out << "float weights_l" << layerIdx << "[num_weights_l" << layerIdx << "] = {";
		for (size_t i = 0; i < weights.size(); i++)
		{
			out << weights[i] << delim;
		}
		out << "};\n";
	}
}


//
//void Parameters::Deserialize(const std::string& inString)
//{
//	std::istringstream input{ inString };
//
//	size_t numBiases;
//	input >> numBiases;
//
//	biases.clear();
//	for (size_t i = 0; i < numBiases; i++)
//	{
//		float bias;
//		input >> bias;
//		biases.push_back(bias);
//	}
//
//	size_t numWeights;
//	input >> numWeights;
//
//	weights.clear();
//	for (size_t i = 0; i < numWeights; i++)
//	{
//		float weight;
//		input >> weight;
//		weights.push_back(weight);
//	}
//
//	std::string end;
//	input >> end;
//
//	assert(end == "end");
//}