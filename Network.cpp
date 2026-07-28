#include "Network.h"
#include "Layer.h"
#include "ActFuncDataBase.h"
#include "iostream"

Network::Network(const std::vector<LayerInfo>& layerInfos, CostFunc::Base* costFunc, bool zeroInit)
{
	m_costFunction = (costFunc != nullptr) ? costFunc : CostFunc::DataBase::FindCostFunc<CostFunc::L1>();

	m_layers.push_back(std::make_unique<InitialLayer>(layerInfos[0].numNeurons));
	m_storedDelta.push_back(Parameters{ m_layers.front()->GetNumNeurons(), m_layers.front()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::Empty>(), 0 });
	m_costDeltas.push_back(std::vector<float>(layerInfos[0].numNeurons, 0.0f));

	for (size_t i = 1; i < layerInfos.size(); i++)
	{
		ActFunc::Base* actFunc = layerInfos[i].actFunc ? layerInfos[i].actFunc : ActFunc::DataBase::FindActFunc<ActFunc::LeakyReLU>();
		m_layers.push_back(std::make_unique<Layer>(layerInfos[i].numNeurons, actFunc, i, m_layers.back().get()));
		m_storedDelta.push_back(Parameters{ m_layers.back()->GetNumNeurons(), m_layers.back()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::Empty>(), i });
		m_costDeltas.push_back(std::vector<float>(layerInfos[i].numNeurons, 0.0f));
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
	if (m_layers.size() == m_storedDelta.size() && m_numStored > 0)
	{
		for (size_t i = 1; i < m_layers.size(); i++)
		{
			const double lr = learningRate * m_layers[i]->m_activationFunction->GetLearningRateMultiplier();
			if (m_layers[i]->m_params.biases.size() == m_storedDelta[i].biases.size()
				&& m_layers[i]->m_params.weights.size() == m_storedDelta[i].weights.size())
			{
				// average = divide by numStored
				// apply learning rate
				// negative because we want to substract. (inverse of the gradient)

				//m_storedDelta[i] *= -1.0f * (lr / m_numStored);
				m_layers[i]->m_params.Add(m_storedDelta[i], lr/ m_numStored);
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
	}
}

void Network::Deserialize(std::istream& in)
{
	std::vector<Parameters> loadedParams(m_layers.size());
	for (size_t i = 0; i < m_layers.size(); ++i)
	{
		loadedParams[i].Deserialize(in);
		if (in.fail() && !in.eof() && i < m_layers.size() - 1)
		{
			std::cout << "[Checkpoint Warning] Failed reading checkpoint stream at layer " << i << ". Retaining new weights.\n";
			return;
		}

		size_t expectedBiases = m_layers[i]->GetNumNeurons();
		size_t expectedWeights = m_layers[i]->GetNumWeightsToPrevious();

		if (loadedParams[i].biases.size() != expectedBiases || loadedParams[i].weights.size() != expectedWeights)
		{
			std::cout << "[Checkpoint Mismatch] Checkpoint layer " << i << " dimensions do not match current network topology!\n"
				<< "  - File:    biases=" << loadedParams[i].biases.size() << ", weights=" << loadedParams[i].weights.size() << "\n"
				<< "  - Network: biases=" << expectedBiases << ", weights=" << expectedWeights << "\n"
				<< "Discarding loaded checkpoint; initializing fresh random weights.\n";
			return;
		}
	}

	for (size_t i = 0; i < m_layers.size(); ++i)
	{
		m_layers[i]->SetParams(loadedParams[i]);
	}
}

float Network::CalculateCost(const std::vector<float>& inputActivation,const std::vector<float>& preferredOutput)
{
	const std::vector<float>& result = Propagate(inputActivation);

	float cost = 0.0f;
	if (result.size() == preferredOutput.size())
	{
		for (size_t i = 0; i < result.size(); i++)
		{
			cost += m_costFunction->Execute(result[i], preferredOutput[i]);
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

const std::vector<float>& Network::PropagateThreadSafe(const std::vector<float>& inputActivation, std::vector<std::vector<float>>& threadBuffers) const
{
	if (threadBuffers.size() != m_layers.size())
	{
		threadBuffers.resize(m_layers.size());
		for (size_t i = 0; i < m_layers.size(); ++i)
		{
			threadBuffers[i].resize(m_layers[i]->GetNumNeurons());
		}
	}

	std::copy(inputActivation.begin(), inputActivation.end(), threadBuffers[0].begin());

	for (size_t l = 1; l < m_layers.size(); ++l)
	{
		const Layer* layer = m_layers[l].get();
		const std::vector<float>& prevAct = threadBuffers[l - 1];
		std::vector<float>& currentAct = threadBuffers[l];

		const size_t numNeurons = layer->GetNumNeurons();
		const size_t weightsPerNeuron = prevAct.size();
		const auto& weights = layer->GetParams().weights;
		const auto& biases = layer->GetParams().biases;
		ActFunc::Base* func = layer->GetActivationFunction();

		for (size_t i = 0; i < numNeurons; ++i)
		{
			float z = biases[i];
			size_t weightStart = i * weightsPerNeuron;
			for (size_t j = 0; j < weightsPerNeuron; ++j)
			{
				z += weights[weightStart + j] * prevAct[j];
			}

			currentAct[i] = func ? func->Execute(z) : z;
		}
	}

	return threadBuffers.back();
}

float Network::BackPropagate(const std::vector<float>& inputActivation, const std::vector<float>& preferredOutput)
{
	// propagate forwards
	float cost = CalculateCost(inputActivation, preferredOutput);

	// create empty network with same dimensions to store deltas. 
	// propagate backwards
	Layer* layer = m_layers.back().get();
	size_t layerIndex = m_layers.size() - 1;

	std::vector<float>& prevLayerCostDeltas = m_costDeltas[layerIndex];
	for (size_t i = 0; i < layer->m_numNeurons; i++)
	{
		float act = layer->m_activations[i];
		float y = preferredOutput[i];

		// the derivative of the cost function
		// a.k.a direction to push our activation to decrease cost ?
		prevLayerCostDeltas[i] = m_costFunction->ExecuteDerivative(act, y);
	}

	while (layer->m_previousLayer != nullptr)
	{
		Parameters& deltaLayer = m_storedDelta[layerIndex];
		const std::vector<float>& currentCostDeltas = m_costDeltas[layerIndex];

		for (size_t i = 0; i < layer->m_numNeurons; i++)
		{
			//calc bias nudge
			float z = layer->m_preProcessedActivations[i];
			float actFuncDeriv = layer->m_activationFunction->ExecuteDerivative(z);

			deltaLayer.biases[i] += actFuncDeriv * currentCostDeltas[i];

			//calc weight nudge (bias nudge * A(L-1)) for each weight
			size_t weightsPerNeuron = layer->m_previousLayer->m_numNeurons;
			size_t weightIndexStart = i * weightsPerNeuron;

			// this because we have a weight for each neuron in the previous layer,
			// might not always be the case.
			for (size_t j = 0; j < layer->m_previousLayer->m_numNeurons; j++)
			{
				deltaLayer.weights[weightIndexStart + j] += layer->m_previousLayer->m_activations[j] * actFuncDeriv * currentCostDeltas[i];
			}
		}

		// setup for next layer
		if (layerIndex > 1)
		{
			std::vector<float>& nextCostDeltas = m_costDeltas[layerIndex - 1];
			std::fill(nextCostDeltas.begin(), nextCostDeltas.end(), 0.0f);

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
					float z = layer->m_preProcessedActivations[j];
					float actFuncDeriv = layer->m_activationFunction->ExecuteDerivative(z);
					float biasDelta = actFuncDeriv * currentCostDeltas[j];
					nextCostDeltas[i] += layer->m_params.weights[weightIdx] * biasDelta; // deltaLayer.biases[j] == actFuncDeriv * currentCostDeltas[i];
				}
			}
		}

		layer = layer->m_previousLayer;
		layerIndex--;
	}

	// update weights and biases
	m_numStored++;

	return cost;
}