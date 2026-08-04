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

	static std::random_device rd;
	//static std::mt19937 generator(rd());
	static std::mt19937 generator(42); // Fixed seed for debugging
	size_t fanIn = (numBiases > 0) ? (numWeights / numBiases) : 1;
	size_t fanOut = numBiases;
	
	for (size_t i = 0; i < numBiases; i++)
	{
		biases[i] = actFunc->GenerateInitialBiases( generator, fanIn, fanOut, lIdx );
	}

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

Parameters& Parameters::operator*=(engineFloat other)
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
    // 1. Load Biases
    size_t numBiases = 0;
    if (in >> numBiases)
    {
       biases.resize(numBiases);
       for (size_t i = 0; i < numBiases; ++i) in >> biases[i];
    }

    // 2. Load Weights
    size_t numWeights = 0;
    if (in >> numWeights)
    {
       weights.resize(numWeights);
       for (size_t i = 0; i < numWeights; ++i) in >> weights[i];
    }

    // 3. Load Adam State (with safety fallback for old checkpoints)
    size_t numMWeights = 0;
    if (in >> numMWeights)
    {
        m_weights.resize(numMWeights);
        for (size_t i = 0; i < numMWeights; ++i) in >> m_weights[i];
    } else {
        m_weights.assign(weights.size(), 0.0f); // Fallback to 0
    }

    size_t numVWeights = 0;
    if (in >> numVWeights)
    {
        v_weights.resize(numVWeights);
        for (size_t i = 0; i < numVWeights; ++i) in >> v_weights[i];
    } else {
        v_weights.assign(weights.size(), 0.0f); // Fallback to 0
    }

    size_t numMBiases = 0;
    if (in >> numMBiases)
    {
        m_biases.resize(numMBiases);
        for (size_t i = 0; i < numMBiases; ++i) in >> m_biases[i];
    } else {
        m_biases.assign(biases.size(), 0.0f); // Fallback to 0
    }

    size_t numVBiases = 0;
    if (in >> numVBiases)
    {
        v_biases.resize(numVBiases);
        for (size_t i = 0; i < numVBiases; ++i) in >> v_biases[i];
    } else {
        v_biases.assign(biases.size(), 0.0f); // Fallback to 0
    }
}

void Parameters::Serialize(std::ostream& out)
{
    // 1. Save Biases
    out << biases.size() << "\n";
    for (size_t i = 0; i < biases.size(); i++)
    {
       out << biases[i] << (i + 1 == biases.size() ? "" : " ");
    }
    out << "\n";

    // 2. Save Weights
    out << weights.size() << "\n";
    for (size_t i = 0; i < weights.size(); i++)
    {
       out << weights[i] << (i + 1 == weights.size() ? "" : " ");
    }
    out << "\n";

    // 3. Save Adam Momentum & Variance for Weights
    out << m_weights.size() << "\n";
    for (size_t i = 0; i < m_weights.size(); i++)
    {
       out << m_weights[i] << (i + 1 == m_weights.size() ? "" : " ");
    }
    out << "\n";

    out << v_weights.size() << "\n";
    for (size_t i = 0; i < v_weights.size(); i++)
    {
       out << v_weights[i] << (i + 1 == v_weights.size() ? "" : " ");
    }
    out << "\n";

    // 4. Save Adam Momentum & Variance for Biases
    out << m_biases.size() << "\n";
    for (size_t i = 0; i < m_biases.size(); i++)
    {
       out << m_biases[i] << (i + 1 == m_biases.size() ? "" : " ");
    }
    out << "\n";

    out << v_biases.size() << "\n";
    for (size_t i = 0; i < v_biases.size(); i++)
    {
       out << v_biases[i] << (i + 1 == v_biases.size() ? "" : " ");
    }
    out << "\n";
}

void Parameters::ApplyAdamUpdate(const std::vector<engineFloat>& gradWeights, const std::vector<engineFloat>& gradBiases, engineFloat learningRate)
{
    // Adam hyperparameters (Standard defaults)
    constexpr engineFloat beta1 = 0.9f;
    constexpr engineFloat beta2 = 0.999f;
    constexpr engineFloat epsilon = 1e-8f;

    adam_t++; // Increment time step

    // Precalculate bias correction denominators
    const engineFloat correction1 = 1.0f - std::pow(beta1, adam_t);
    const engineFloat correction2 = 1.0f - std::pow(beta2, adam_t);

    // Update Weights
    for (size_t i = 0; i < weights.size(); i++)
    {
        engineFloat g = gradWeights[i];
        m_weights[i] = beta1 * m_weights[i] + (1.0f - beta1) * g;
        v_weights[i] = beta2 * v_weights[i] + (1.0f - beta2) * (g * g);

        engineFloat m_hat = m_weights[i] / correction1;
        engineFloat v_hat = v_weights[i] / correction2;

        weights[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }

    // Update Biases
    for (size_t i = 0; i < biases.size(); i++)
    {
        engineFloat g = gradBiases[i];
        m_biases[i] = beta1 * m_biases[i] + (1.0f - beta1) * g;
        v_biases[i] = beta2 * v_biases[i] + (1.0f - beta2) * (g * g);

        engineFloat m_hat = m_biases[i] / correction1;
        engineFloat v_hat = v_biases[i] / correction2;

        biases[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }
}