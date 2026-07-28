#pragma once
#include <vector>
#include "Serializable.h"
#include "ActFuncDataBase.h"

struct Parameters : public Serializable
{
	Parameters() = default;
	Parameters(size_t numBiases, size_t numWeights, ActFunc::Base* actFunc, size_t layerIdx);
	void Update(const Parameters& gradient, double learningRate);
	Parameters& operator*=(float other);
	Parameters& operator+=(const Parameters& other);
	
	void Clear();
	std::vector<float> w_velocities;
	std::vector<float> b_velocities;

	std::vector<float> weights;
	std::vector<float> biases;

	size_t layerIdx;

	// Inherited via Serializable
	virtual void Serialize(std::ostream& out) override;
	virtual void Deserialize(std::istream& in) override;
};

