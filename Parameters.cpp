#include "Parameters.h"
#include <sstream>
#include <cassert>
#include <cmath> // For sqrtf


Parameters::Parameters(size_t numBiases, size_t numWeights, ActFunc::Base* actFunc, size_t lIdx)
{
	biases.resize(numBiases, 0.0f);
	weights.resize(numWeights, 0.0f);

	// Initialize Adam vectors to 0
    m_biases.resize(numBiases, 0.0f);
    v_biases.resize(numBiases, 0.0f);
    m_weights.resize(numWeights, 0.0f);
    v_weights.resize(numWeights, 0.0f);
    adam_t = 0;

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
		weights[i] = actFunc->GenerateInitialWeight( generator, fanIn, fanOut, lIdx );
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

void Parameters::Deserialize(std::istream& in)
{
	size_t numBiases = 0;
	if (in >> numBiases)
	{
		biases.resize(numBiases);
		for (size_t i = 0; i < numBiases; ++i)
		{
			in >> biases[i];
		}
	}

	size_t numWeights = 0;
	if (in >> numWeights)
	{
		weights.resize(numWeights);
		for (size_t i = 0; i < numWeights; ++i)
		{
			in >> weights[i];
		}
	}
}

void Parameters::Serialize(std::ostream& out)
{
	out << biases.size() << "\n";
	for (size_t i = 0; i < biases.size(); i++)
	{
		out << biases[i] << (i + 1 == biases.size() ? "" : " ");
	}
	out << "\n";

	out << weights.size() << "\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		out << weights[i] << (i + 1 == weights.size() ? "" : " ");
	}
	out << "\n";
}

void Parameters::ApplyAdamUpdate(const std::vector<float>& gradWeights, const std::vector<float>& gradBiases, float learningRate)
{
    // Adam hyperparameters (Standard defaults)
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    adam_t++; // Increment time step

    // Precalculate bias correction denominators
    float correction1 = 1.0f - std::pow(beta1, adam_t);
    float correction2 = 1.0f - std::pow(beta2, adam_t);

    // Update Weights
    for (size_t i = 0; i < weights.size(); i++)
    {
        float g = gradWeights[i];
        m_weights[i] = beta1 * m_weights[i] + (1.0f - beta1) * g;
        v_weights[i] = beta2 * v_weights[i] + (1.0f - beta2) * (g * g);

        float m_hat = m_weights[i] / correction1;
        float v_hat = v_weights[i] / correction2;

        weights[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }

    // Update Biases
    for (size_t i = 0; i < biases.size(); i++)
    {
        float g = gradBiases[i];
        m_biases[i] = beta1 * m_biases[i] + (1.0f - beta1) * g;
        v_biases[i] = beta2 * v_biases[i] + (1.0f - beta2) * (g * g);

        float m_hat = m_biases[i] / correction1;
        float v_hat = v_biases[i] / correction2;

        biases[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }
}