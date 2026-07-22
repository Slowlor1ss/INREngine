#pragma once
#include <vector>
#include <memory>
#include "Parameters.h"
#include "ActivationFunctions.h"

class Layer : public Serializable
{
public:
	Layer(size_t numNeurons, ActFunc::Base* func, size_t layerIdx, Layer* previousLayer = nullptr);
	virtual ~Layer() = default;

	void SetParams(const Parameters& params);

	size_t GetNumNeurons() const
	{
		return m_numNeurons;
	}

	size_t GetNumWeightsToPrevious() const
	{
		return m_params.weights.size();
	}

	// Inherited via Serializable
	virtual void Serialize(std::ostream& out) override;
	virtual void Deserialize(std::istream& in) override {};

protected:
	void Propagate();

protected:
	ActFunc::Base* m_activationFunction;

	Layer* m_nextLayer = nullptr;
	Layer* m_previousLayer = nullptr;

	size_t m_layerIdx;

	Parameters m_params;

	size_t m_numNeurons;

	// actiavtion before the activationfunction applies
	std::vector<float> m_preProcessedActivations;

	// final activation
	std::vector<float> m_activations;

	friend class Network;
};

class InitialLayer : public Layer
{
public:
	InitialLayer(size_t numNeurons);
	virtual ~InitialLayer() = default;
	void StartPropagation(std::vector<float> inputActivation);
};
