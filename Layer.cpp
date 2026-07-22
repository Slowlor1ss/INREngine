#include "Layer.h"

Layer::Layer(size_t numNeurons, ActFunc::Base* func, size_t layerIdx, Layer* previousLayer)
	: m_numNeurons{numNeurons}
	, m_activationFunction{func}
	, m_previousLayer{previousLayer}
	, m_layerIdx{layerIdx}
{
	m_preProcessedActivations.resize(numNeurons);
	m_activations.resize(numNeurons);

	size_t numWeightsToEachNeuron = 0;
	if (previousLayer)
	{
		previousLayer->m_nextLayer = this;
		numWeightsToEachNeuron = previousLayer->m_numNeurons;
	}

	m_params = Parameters{ numNeurons, numWeightsToEachNeuron * numNeurons, func, layerIdx };
}

void Layer::SetParams(const Parameters& params)
{
	// check num neurons and weights
	m_params = params;
}

void Layer::Serialize(std::ostream& out)
{
	m_params.Serialize(out);
}

void Layer::Propagate()
{
	// We need previousLayer to work
	if (m_previousLayer != nullptr)
	{
		// For each of my neurons
		size_t weightsPerNeuron = m_previousLayer->m_activations.size();
		for (size_t i = 0; i < m_activations.size(); i++)
		{
			size_t startWeight = i * weightsPerNeuron;

			float activation = 0.0f;
			for (size_t j = 0; j < m_previousLayer->m_activations.size(); j++)
			{
				float prevActivation = m_previousLayer->m_activations[j];
				float weight = m_params.weights[startWeight + j];
				activation += weight * prevActivation;
			}
			activation += m_params.biases[i];

			m_preProcessedActivations[i] = activation;
			if (m_activationFunction != nullptr)
			{
				activation = m_activationFunction->Execute(activation);
			}

			m_activations[i] = activation;
		}
	}

	// propagate
	if (m_nextLayer)
	{
		m_nextLayer->Propagate();
	}
}

InitialLayer::InitialLayer(size_t numNeurons)
	:Layer{numNeurons, ActFunc::DataBase::FindActFunc<ActFunc::None>(), 0, nullptr}
{
}

void InitialLayer::StartPropagation(std::vector<float> inputActivation)
{
	if (inputActivation.size() == m_numNeurons)
	{
		m_activations = std::move(inputActivation);
		m_preProcessedActivations = m_activations;
		Propagate();
	}
}
