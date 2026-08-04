#pragma once
#include <vector>
#include "Serializable.h"
#include "ActFuncDataBase.h"
#include "Types.h"

struct Parameters : public Serializable
{
	Parameters() = default;
	Parameters(size_t numBiases, size_t numWeights, ActFunc::Base* actFunc, size_t layerIdx);
	Parameters& operator+=(const Parameters& other);
	Parameters& operator*=(engineFloat other);
	
	void Clear();
	std::vector<engineFloat> weights; // Position (actual real weights)
	std::vector<engineFloat> biases;

	size_t layerIdx;

    // Adam state variables
    std::vector<engineFloat> m_weights; // Momentum
    std::vector<engineFloat> v_weights; // Velocity
    std::vector<engineFloat> m_biases;
    std::vector<engineFloat> v_biases;
    
    int adam_t = 0; // Time step counter for bias correction

	// Inherited via Serializable
	virtual void Serialize(std::ostream& out) override;
	void ApplyAdamUpdate(const std::vector<engineFloat>& gradWeights, const std::vector<engineFloat>& gradBiases,
	                     engineFloat learningRate);
	virtual void Deserialize(std::istream& in) override;
};

