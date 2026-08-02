#pragma once
#include <vector>
#include "Serializable.h"
#include "ActFuncDataBase.h"

struct Parameters : public Serializable
{
	Parameters() = default;
	Parameters(size_t numBiases, size_t numWeights, ActFunc::Base* actFunc, size_t layerIdx);
	Parameters& operator+=(const Parameters& other);
	Parameters& operator*=(float other);
	
	void Clear();
	std::vector<float> weights;
	std::vector<float> biases;

	size_t layerIdx;

    // Adam state variables
    std::vector<float> m_weights;
    std::vector<float> v_weights;
    std::vector<float> m_biases;
    std::vector<float> v_biases;
    
    int adam_t = 0; // Time step counter for bias correction

	// Inherited via Serializable
	virtual void Serialize(std::ostream& out) override;
	void ApplyAdamUpdate(const std::vector<float>& gradWeights, const std::vector<float>& gradBiases,
	                     float learningRate);
	virtual void Deserialize(std::istream& in) override;
};

