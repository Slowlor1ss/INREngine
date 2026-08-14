#include "Parameters.h"
#include <algorithm>
#include <sstream>
#include <cassert>
#include <cmath> // For sqrtf


Parameters::Parameters(size_t numBiases, size_t numWeights, ActFunc::Base* actFunc, size_t lIdx)
{
	layerIdx = lIdx;
    adam_t = 0;
	
	biases.resize(numBiases, 0.0f);
	weights.resize(numWeights, 0.0f);

	// Initialize Adam vectors to 0
    m_biases.resize(numBiases, 0.0f);
    v_biases.resize(numBiases, 0.0f);
    m_weights.resize(numWeights, 0.0f);
    v_weights.resize(numWeights, 0.0f);

	static std::random_device rd;
	//static std::mt19937 generator(rd());
	// TODO: make teh generator in some utility header of PCH and replace everywhere
	static std::mt19937 generator(42); // Fixed seed for debugging
	const size_t fanIn = (numBiases > 0) ? (numWeights / numBiases) : 1;
	const size_t fanOut = numBiases;
	
	bool isDualWeights = actFunc->GetName() == "Wire" || actFunc->GetName() == "WireHybrid";
	if (isDualWeights)
	{
		// Allocate Secondary Vectors ONLY for Wire
		weights_scale.resize(numWeights, 0.0f);
		biases_scale.resize(numBiases, 0.0f);
        
		m_weights_scale.resize(numWeights, 0.0f);
		v_weights_scale.resize(numWeights, 0.0f);
		m_biases_scale.resize(numBiases, 0.0f);
		v_biases_scale.resize(numBiases, 0.0f);
	}
	
	for (size_t i = 0; i < numBiases; i++)
	{
		biases[i] = actFunc->GenerateInitialBiases( generator, fanIn, fanOut, lIdx );
		if (isDualWeights)
		{
			biases_scale[i] = actFunc->GenerateInitialBiases( generator, fanIn, fanOut, lIdx );
		}
	}

	for (size_t i = 0; i < numWeights; i++)
	{
		weights[i] = actFunc->GenerateInitialWeight( generator, fanIn, fanOut, lIdx );
		if (isDualWeights)
		{
			weights_scale[i] = actFunc->GenerateInitialWeight( generator, fanIn, fanOut, lIdx );
		}
	}
	
	UploadToGPU();
}

// Note: this does not chnage the adams state paraeters!
Parameters& Parameters::operator+=(const Parameters& other)
{
	// Base
	if (biases.size() == other.biases.size())
	{
		for (size_t j = 0; j < biases.size(); j++) biases[j] += other.biases[j];
	}
	if (weights.size() == other.weights.size())
	{
		for (size_t j = 0; j < weights.size(); j++) weights[j] += other.weights[j];
	}
    
	// Dual-Weight Arrays
	if (biases_scale.size() == other.biases_scale.size())
	{
		for (size_t j = 0; j < biases_scale.size(); j++) biases_scale[j] += other.biases_scale[j];
	}
	if (weights_scale.size() == other.weights_scale.size())
	{
		for (size_t j = 0; j < weights_scale.size(); j++) weights_scale[j] += other.weights_scale[j];
	}

	return *this;
}

// Note: this does not chnage the adams state paraeters!
Parameters& Parameters::operator*=(engineFloat other)
{
	// Base
	for (float& biase : biases) biase *= other;
	for (float& weight : weights) weight *= other;
    
	// Dual-Weight Arrays
	for (float& j : biases_scale) j *= other;
	for (float& j : weights_scale) j *= other;

	return *this;
}

Parameters Parameters::Clone() const {
	Parameters p;
	p.layerIdx = layerIdx;
	p.adam_t = adam_t;

	p.weights = weights;
	p.biases = biases;
	p.weights_scale = weights_scale;
	p.biases_scale = biases_scale;

	p.m_weights = m_weights;
	p.v_weights = v_weights;
	p.m_biases = m_biases;
	p.v_biases = v_biases;

	p.m_weights_scale = m_weights_scale;
	p.v_weights_scale = v_weights_scale;
	p.m_biases_scale = m_biases_scale;
	p.v_biases_scale = v_biases_scale;

	p.UploadToGPU();
	return p;
}

void Parameters::Clear()
{
	// Zero out parameters
	std::ranges::fill(weights, 0.0f);
	std::ranges::fill(biases, 0.0f);
	std::ranges::fill(weights_scale, 0.0f);
	std::ranges::fill(biases_scale, 0.0f);

	// Zero out Adam states
	std::ranges::fill(m_weights, 0.0f);
	std::ranges::fill(v_weights, 0.0f);
	std::ranges::fill(m_biases, 0.0f);
	std::ranges::fill(v_biases, 0.0f);
    
	std::ranges::fill(m_weights_scale, 0.0f);
	std::ranges::fill(v_weights_scale, 0.0f);
	std::ranges::fill(m_biases_scale, 0.0f);
	std::ranges::fill(v_biases_scale, 0.0f);

	// Reset Adam time step
	adam_t = 0;
}

void Parameters::Deserialize(std::istream& in)
{
    // Load Biases
    size_t numBiases = 0;
    if (in >> numBiases)
    {
       biases.resize(numBiases);
       for (size_t i = 0; i < numBiases; ++i) in >> biases[i];
    }

    // Load Weights
    size_t numWeights = 0;
    if (in >> numWeights)
    {
       weights.resize(numWeights);
       for (size_t i = 0; i < numWeights; ++i) in >> weights[i];
    }

    // Load Adam State (with safety fallback for old checkpoints)
    size_t numMWeights = 0;
    if (in >> numMWeights)
    {
        m_weights.resize(numMWeights);
        for (size_t i = 0; i < numMWeights; ++i) in >> m_weights[i];
    }

    size_t numVWeights = 0;
    if (in >> numVWeights)
    {
        v_weights.resize(numVWeights);
        for (size_t i = 0; i < numVWeights; ++i) in >> v_weights[i];
    }

    size_t numMBiases = 0;
    if (in >> numMBiases)
    {
        m_biases.resize(numMBiases);
        for (size_t i = 0; i < numMBiases; ++i) in >> m_biases[i];
    }

    size_t numVBiases = 0;
    if (in >> numVBiases)
    {
        v_biases.resize(numVBiases);
        for (size_t i = 0; i < numVBiases; ++i) in >> v_biases[i];
    }
	
	// Load Secondary (Scale) Weights and Biases (Safe fallback for old checkpoints)
	size_t numWeightsScale = 0;
	if (in >> numWeightsScale)
	{
		weights_scale.resize(numWeightsScale);
		for (size_t i = 0; i < numWeightsScale; ++i) in >> weights_scale[i];
	}

	size_t numBiasesScale = 0;
	if (in >> numBiasesScale)
	{
		biases_scale.resize(numBiasesScale);
		for (size_t i = 0; i < numBiasesScale; ++i) in >> biases_scale[i];
	}

	// Load Secondary Adam States
	size_t numMWeightsScale = 0;
	if (in >> numMWeightsScale)
	{
		m_weights_scale.resize(numMWeightsScale);
		for (size_t i = 0; i < numMWeightsScale; ++i) in >> m_weights_scale[i];
	}

	size_t numVWeightsScale = 0;
	if (in >> numVWeightsScale)
	{
		v_weights_scale.resize(numVWeightsScale);
		for (size_t i = 0; i < numVWeightsScale; ++i) in >> v_weights_scale[i];
	}

	size_t numMBiasesScale = 0;
	if (in >> numMBiasesScale)
	{
		m_biases_scale.resize(numMBiasesScale);
		for (size_t i = 0; i < numMBiasesScale; ++i) in >> m_biases_scale[i];
	}

	size_t numVBiasesScale = 0;
	if (in >> numVBiasesScale)
	{
		v_biases_scale.resize(numVBiasesScale);
		for (size_t i = 0; i < numVBiasesScale; ++i) in >> v_biases_scale[i];
	}
	
	UploadToGPU();
}

void Parameters::Serialize(std::ostream& out)
{
	ReadbackFromGPU();
	
    // Save Biases
    out << biases.size() << "\n";
    for (size_t i = 0; i < biases.size(); i++)
    {
       out << biases[i] << (i + 1 == biases.size() ? "" : " ");
    }
    out << "\n";

    // Save Weights
    out << weights.size() << "\n";
    for (size_t i = 0; i < weights.size(); i++)
    {
       out << weights[i] << (i + 1 == weights.size() ? "" : " ");
    }
    out << "\n";

    // Save Adam Momentum & Variance for Weights
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

    // Save Adam Momentum & Variance for Biases
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
	
	// Save Secondary (Scale) Weights and Biases
	out << weights_scale.size() << "\n";
	for (size_t i = 0; i < weights_scale.size(); i++)
	{
		out << weights_scale[i] << (i + 1 == weights_scale.size() ? "" : " ");
	}
	out << "\n";

	out << biases_scale.size() << "\n";
	for (size_t i = 0; i < biases_scale.size(); i++)
	{
		out << biases_scale[i] << (i + 1 == biases_scale.size() ? "" : " ");
	}
	out << "\n";

	// Save Secondary Adam States
	out << m_weights_scale.size() << "\n";
	for (size_t i = 0; i < m_weights_scale.size(); i++)
	{
		out << m_weights_scale[i] << (i + 1 == m_weights_scale.size() ? "" : " ");
	}
	out << "\n";

	out << v_weights_scale.size() << "\n";
	for (size_t i = 0; i < v_weights_scale.size(); i++)
	{
		out << v_weights_scale[i] << (i + 1 == v_weights_scale.size() ? "" : " ");
	}
	out << "\n";

	out << m_biases_scale.size() << "\n";
	for (size_t i = 0; i < m_biases_scale.size(); i++)
	{
		out << m_biases_scale[i] << (i + 1 == m_biases_scale.size() ? "" : " ");
	}
	out << "\n";

	out << v_biases_scale.size() << "\n";
	for (size_t i = 0; i < v_biases_scale.size(); i++)
	{
		out << v_biases_scale[i] << (i + 1 == v_biases_scale.size() ? "" : " ");
	}
	out << "\n";
}

void Parameters::ApplyAdamUpdate(
	const std::vector<engineFloat>& gradWeights, 
	const std::vector<engineFloat>& gradBiases,
	const std::vector<engineFloat>& gradWeightsScale, 
	const std::vector<engineFloat>& gradBiasesScale,
	engineFloat learningRate)
{
    // Adam hyperparameters (Standard defaults)
    constexpr engineFloat beta1 = 0.9f;
    constexpr engineFloat beta2 = 0.999f;
    constexpr engineFloat epsilon = std::numeric_limits<engineFloat>::epsilon();//1e-8f;

    adam_t++; // Increment time step

    // Precalculate bias correction denominators
    const engineFloat correction1 = 1.0f - std::pow(beta1, static_cast<engineFloat>(adam_t));
    const engineFloat correction2 = 1.0f - std::pow(beta2, static_cast<engineFloat>(adam_t));

    // Update Weights
    for (size_t i = 0; i < weights.size(); i++)
    {
        const engineFloat g = gradWeights[i];
        m_weights[i] = beta1 * m_weights[i] + (1.0f - beta1) * g;
        v_weights[i] = beta2 * v_weights[i] + (1.0f - beta2) * (g * g);

        const engineFloat m_hat = m_weights[i] / correction1;
        const engineFloat v_hat = v_weights[i] / correction2;

        weights[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }

    // Update Biases
    for (size_t i = 0; i < biases.size(); i++)
    {
        const engineFloat g = gradBiases[i];
        m_biases[i] = beta1 * m_biases[i] + (1.0f - beta1) * g;
        v_biases[i] = beta2 * v_biases[i] + (1.0f - beta2) * (g * g);

        const engineFloat m_hat = m_biases[i] / correction1;
        const engineFloat v_hat = v_biases[i] / correction2;

        biases[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }
	
	if (!weights_scale.empty() && !gradWeightsScale.empty())
	{
		for (size_t i = 0; i < weights_scale.size(); i++)
		{
			const engineFloat g = gradWeightsScale[i];
			m_weights_scale[i] = beta1 * m_weights_scale[i] + (1.0f - beta1) * g;
			v_weights_scale[i] = beta2 * v_weights_scale[i] + (1.0f - beta2) * (g * g);

			const engineFloat m_hat = m_weights_scale[i] / correction1;
			const engineFloat v_hat = v_weights_scale[i] / correction2;

			weights_scale[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
		}

		for (size_t i = 0; i < biases_scale.size(); i++)
		{
			const engineFloat g = gradBiasesScale[i];
			m_biases_scale[i] = beta1 * m_biases_scale[i] + (1.0f - beta1) * g;
			v_biases_scale[i] = beta2 * v_biases_scale[i] + (1.0f - beta2) * (g * g);

			const engineFloat m_hat = m_biases_scale[i] / correction1;
			const engineFloat v_hat = v_biases_scale[i] / correction2;

			biases_scale[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
		}
	}
}

void Parameters::UploadToGPU()
{
	d_weights.Upload(weights);
	d_biases.Upload(biases);
	if (!weights_scale.empty()) d_weights_scale.Upload(weights_scale);
	if (!biases_scale.empty()) d_biases_scale.Upload(biases_scale);
}

void Parameters::ReadbackFromGPU()
{
	d_weights.Readback(weights);
	d_biases.Readback(biases);
	if (!weights_scale.empty()) d_weights_scale.Readback(weights_scale);
	if (!biases_scale.empty()) d_biases_scale.Readback(biases_scale);
}
