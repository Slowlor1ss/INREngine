#pragma once
#include <vector>
#include "Serializable.h"
#include "ActFuncDataBase.h"
#include "GpuBuffer.cuh"
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
	
	// GPU shadow copy
	GpuBuffer<engineFloat> d_weights;
	GpuBuffer<engineFloat> d_biases;
	
	// Secondary Parameters (Scale/Envelope for Dual-Weight WIRE)
	std::vector<engineFloat> weights_scale;
	std::vector<engineFloat> biases_scale;
	
	GpuBuffer<engineFloat> d_weights_scale;
	GpuBuffer<engineFloat> d_biases_scale;

	size_t layerIdx;

    // Adam state variables
    std::vector<engineFloat> m_weights; // Momentum
    std::vector<engineFloat> v_weights; // Velocity
    std::vector<engineFloat> m_biases;
    std::vector<engineFloat> v_biases;
	
	// Secondary Adam States
	std::vector<engineFloat> m_weights_scale;
	std::vector<engineFloat> v_weights_scale;
	std::vector<engineFloat> m_biases_scale;
	std::vector<engineFloat> v_biases_scale;
    
    int adam_t = 0; // Time step counter for bias correction

	// Inherited via Serializable
	virtual void Serialize(std::ostream& out) override;
	void ApplyAdamUpdate(
		const std::vector<engineFloat>& gradWeights, 
		const std::vector<engineFloat>& gradBiases,
		const std::vector<engineFloat>& gradWeightsScale, 
		const std::vector<engineFloat>& gradBiasesScale,
		engineFloat learningRate);
	virtual void Deserialize(std::istream& in) override;
	
	void UploadToGPU();
	void ReadbackFromGPU();
};

