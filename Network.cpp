#include "Network.h"
#include "Layer.h"
#include "ActFuncDataBase.h"

Network::Network(const std::vector<size_t>& neuronsPerLayer, bool zeroInit)
{
	m_layers.push_back(std::make_unique<InitialLayer>(neuronsPerLayer[0]));
	m_storedDelta.push_back(Parameters{ m_layers.front()->GetNumNeurons(), m_layers.front()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), 0 });

	for (size_t i = 1; i < neuronsPerLayer.size(); i++)
	{
		ActFunc::Base* actFunc;

		if (i == neuronsPerLayer.size() - 1)
		{
			// could be softmax
			actFunc = ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>();
		}
		else
		{
			actFunc = ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>();
		}

		m_layers.push_back(std::make_unique<Layer>(neuronsPerLayer[i], actFunc, i, m_layers.back().get()));
		m_storedDelta.push_back(Parameters{ m_layers.back()->GetNumNeurons(), m_layers.back()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), i });
	}
}

void Network::StoreDelta(const std::vector<Parameters>& other)
{
	if (m_storedDelta.size() == other.size())
	{
		// no need to update first layer
		for (size_t i = 1; i < m_storedDelta.size(); i++)
		{
			m_storedDelta[i] += other[i];
		}

		m_numStored++;
	}
}

InitialLayer& Network::GetInitialLayer()
{
	return *static_cast<InitialLayer*>(m_layers.front().get());
}

void Network::ConsumeDelta(float learningRate)
{
	if (m_layers.size() == m_storedDelta.size())
	{
		for (size_t i = 1; i < m_layers.size(); i++)
		{
			const float lr = learningRate * m_layers[i]->m_activationFunction->GetLearningRateMultiplier();
			if (m_layers[i]->m_params.biases.size() == m_storedDelta[i].biases.size()
				&& m_layers[i]->m_params.weights.size() == m_storedDelta[i].weights.size())
			{
				// average = divide by numStored
				// apply learning rate
				// negative because we want to substract. (inverse of the gradient)

				m_storedDelta[i] *= -1.0f * (lr / m_numStored);
				m_layers[i]->m_params += m_storedDelta[i];
				m_storedDelta[i].Clear();
			}
		}

		m_numStored = 0;
	}
}

void Network::Serialize(std::ostream& out)
{
	for (const auto& l : m_layers)
	{
		l->Serialize(out);
		out << '\n';
	}
}

float Network::CalculateCost(const std::vector<float>& inputActivation,const std::vector<float>& preferredOutput)
{
	const std::vector<float>& result = Propagate(inputActivation);

	float cost = 0.0f;
	if (result.size() == preferredOutput.size())
	{
		// MSE
		//for (size_t i = 0; i < result.size(); i++)
		//{
		//	cost += powf(result[i] - preferredOutput[i], 2.0f);
		//}
		//return cost;

		// L1 Loss
		for (size_t i = 0; i < result.size(); i++)
		{
			cost += abs(result[i] - preferredOutput[i]);
		}
		return cost / result.size();

	}

	return std::numeric_limits<float>().infinity();
}

std::vector<float> Network::Propagate(const std::vector<float>& inputActivation)
{
	GetInitialLayer().StartPropagation(inputActivation);
	return m_layers.back()->m_activations;
}

float Network::BackPropagate(const std::vector<float>& inputActivation,const std::vector<float>& preferredOutput)
{
	// propagate forwards
	float cost = CalculateCost(inputActivation, preferredOutput);

	// create empty network with same dimensions to store deltas. 
	std::vector<Parameters> deltaParameters;
	deltaParameters.resize(m_layers.size());
	deltaParameters.front() = Parameters{ GetInitialLayer().GetNumNeurons(), GetInitialLayer().GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), 0 };

	// propagate backwards
	Layer* layer = m_layers.back().get();

	std::vector<float> prevLayerCostDeltas;
	prevLayerCostDeltas.resize(layer->m_numNeurons);
	for (size_t i = 0; i < layer->m_numNeurons; i++)
	{

		float act = layer->m_activations[i];
		float y = preferredOutput[i];

		// the derivative of the cost function (act - y)^2
		// a.k.a direction to push our activation to decrease cost ?
		
		// MSE
		prevLayerCostDeltas[i] = 2 * (act - y);

		// derivative of l1 loss cost function
		prevLayerCostDeltas[i] = y > act ? -1.0f : 1.0f;

	}

	size_t layerIndex = m_layers.size() - 1;
	while (layer->m_previousLayer != nullptr)
	{
		Parameters& deltaLayer = deltaParameters[layerIndex] = Parameters(layer->GetNumNeurons(), layer->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), layer->m_layerIdx);

		std::vector<float> currentCostDeltas = prevLayerCostDeltas;

		for (size_t i = 0; i < layer->m_numNeurons; i++)
		{
			//calc bias nudge
			float z = layer->m_preProcessedActivations[i];
			float actFuncDeriv = layer->m_activationFunction->ExecuteDerivative(z);

			deltaLayer.biases[i] = actFuncDeriv * currentCostDeltas[i];

			//calc weight nudge (bias nudge * A(L-1)) for each weight
			size_t weightsPerNeuron = layer->m_previousLayer->m_numNeurons;
			size_t weightIndexStart = i * weightsPerNeuron;

			// this because we have a weight for each neuron in the previous layer,
			// might not always be the case.
			for (size_t j = 0; j < layer->m_previousLayer->m_numNeurons; j++)
			{
				deltaLayer.weights[weightIndexStart + j] = layer->m_previousLayer->m_activations[j] * actFuncDeriv * currentCostDeltas[i];
			}
		}

		// setup for next layer
		prevLayerCostDeltas.clear();
		prevLayerCostDeltas.resize(layer->m_previousLayer->m_numNeurons, 0.0f);

		for (size_t i = 0; i < layer->m_previousLayer->m_numNeurons; i++)
		{
			// for each outgoing weight from neuron[j]
			// calc weight * weightDestNeuron.biasDelta;
			// sum it. save in prevLayerCostDeltas[j]
			size_t relevantWeightIndex = i;
			size_t weightsPerNeuron = layer->m_previousLayer->m_numNeurons;
			for (size_t j = 0; j < layer->m_numNeurons; j++)
			{
				size_t weightIdx = weightsPerNeuron * j + relevantWeightIndex;
				prevLayerCostDeltas[i] += layer->m_params.weights[weightIdx] * deltaLayer.biases[j]; // deltaLayer.biases[j] == actFuncDeriv * currentCostDeltas[i];
			}
		}
		layer = layer->m_previousLayer;
		layerIndex--;
	}

	// update weights and biases
	StoreDelta(deltaParameters);

	return cost;
}