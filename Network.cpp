#include "Network.h"
#include "Layer.h"
#include "ActFuncDataBase.h"
#include "iostream"

Network::Network(
    const std::vector<size_t>& neuronsPerLayer, 
    ActFunc::Base* hiddenActivation, 
    ActFunc::Base* outputActivation, 
    CostFunc::Base* costFunc, 
    bool zeroInit)
{
    m_costFunction = (costFunc != nullptr) ? costFunc : CostFunc::DataBase::FindCostFunc<CostFunc::L1>();

    m_layers.push_back(std::make_unique<InitialLayer>(neuronsPerLayer[0]));
    m_storedDelta.push_back(Parameters{ m_layers.front()->GetNumNeurons(), m_layers.front()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), 0 });
	m_storedDelta.back().Clear(); // TODO: unsure
    m_costDeltas.push_back(std::vector<float>(neuronsPerLayer[0], 0.0f));

    for (size_t i = 1; i < neuronsPerLayer.size(); i++)
    {
        // Choose the activation function based on the layer index
        ActFunc::Base* actFunc = (i == neuronsPerLayer.size() - 1) ? outputActivation : hiddenActivation;

        m_layers.push_back(std::make_unique<Layer>(neuronsPerLayer[i], actFunc, i, m_layers.back().get()));
        m_storedDelta.push_back(Parameters{ m_layers.back()->GetNumNeurons(), m_layers.back()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), i });
		m_storedDelta.back().Clear(); // TODO: unsure
    	m_costDeltas.push_back(std::vector<float>(neuronsPerLayer[i], 0.0f));
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

void Network::BackPropagateGradientGuided(
    const ImageUtils::SpatialData& target,
    const std::vector<float>& input, // TODO: remove unused
    const std::vector<std::vector<float>>& forwardActivations,
    const std::vector<std::vector<float>>& forwardSums,
    const SpatialDerivativeBuffer& spatialBuffers,
    float learningRate)
{
    // ------
    // Calculate the 3 Error Signals at the Output Layer
    // ------
    size_t numLayers = m_layers.size();
    const auto& outputActivations = forwardActivations.back();
    const auto& outputGradX = spatialBuffers.gradientX.back();
    const auto& outputGradY = spatialBuffers.gradientY.back();

    std::vector<float> colorError(outputActivations.size());
    std::vector<float> errorGradX(outputActivations.size());
    std::vector<float> errorGradY(outputActivations.size());

    //for (size_t i = 0; i < outputActivations.size(); ++i) {
    //    // Derivative of (Prediction - Target)^2 is 2 * (Prediction - Target)
    //    colorError[i] = 2.0f * (outputActivations[i] - target.values[i]);
    //    errorGradX[i] = 2.0f * (outputGradX[i] - target.gradX[i]);
    //    errorGradY[i] = 2.0f * (outputGradY[i] - target.gradY[i]);
    //}

	// A hyperparameter to balance how much the network cares about slopes vs colors.
    // 0.01f is a great starting point so the massive slopes don't nuke the colors.
    float spatialLossWeight = 1.f; 

    for (size_t i = 0; i < outputActivations.size(); ++i) {
        colorError[i] = 2.0f * (outputActivations[i] - target.values[i]);
        
        // Scale down the spatial errors!
        errorGradX[i] = 2.0f * (outputGradX[i] - target.gradX[i]) * spatialLossWeight;
        errorGradY[i] = 2.0f * (outputGradY[i] - target.gradY[i]) * spatialLossWeight;
    }

    // ------
    // Update All Layers (Looping from Output down to Layer 0)
    // ------
    for (int l = (int)numLayers - 1; l >= 1; --l) 
    {
        // // Grab the previous layer's activations (or the raw input if it's layer 0)
        // const std::vector<float>& prevActivations = (l == 0) ? input : forwardActivations[l - 1]; 
        // // Grab the previous layer's slopes (or the target's input slopes if layer 0)
        // const std::vector<float>& prevGradX = (l == 0) ? target.gradX : spatialBuffers.gradientX[l - 1]; 
        // const std::vector<float>& prevGradY = (l == 0) ? target.gradY : spatialBuffers.gradientY[l - 1]; 
    	
    	// Grab the previous layer's data directly from the buffers
    	const std::vector<float>& prevActivations = forwardActivations[l - 1]; 
    	const std::vector<float>& prevGradX = spatialBuffers.gradientX[l - 1]; 
    	const std::vector<float>& prevGradY = spatialBuffers.gradientY[l - 1];
    	
        size_t weightsPerNeuron = prevActivations.size();
        
        // Prepare empty error arrays to pass back to the previous layer
        std::vector<float> nextColorError(weightsPerNeuron, 0.0f);
        std::vector<float> nextErrorGradX(weightsPerNeuron, 0.0f);
        std::vector<float> nextErrorGradY(weightsPerNeuron, 0.0f);
    	
        for (size_t i = 0; i < m_layers[l]->GetNumNeurons(); ++i) 
        {
            float z = forwardSums[l][i];
            float d1 = m_layers[l]->m_activationFunction->ExecuteDerivative(z); // f'(z)
            float d2 = m_layers[l]->m_activationFunction->ExecuteSecondDerivative(z); // f''(z)

            size_t startWeight = i * weightsPerNeuron;
            
            // To calculate f''(z) * Sum(w * prev_slope), we need the raw pre-activation slope
            float rawSlopeX = 0.0f;
            float rawSlopeY = 0.0f;
            for (size_t j = 0; j < prevActivations.size(); ++j) {
                float weight = m_layers[l]->m_params.weights[startWeight + j];
                rawSlopeX += weight * prevGradX[j];
                rawSlopeY += weight * prevGradY[j];
            }

            for (size_t j = 0; j < prevActivations.size(); ++j) 
            {
                float prevA = prevActivations[j];
                float oldWeight = m_layers[l]->m_params.weights[startWeight + j]; // Save BEFORE modifying!

                // How this weight affects the Color Error
                float gradColor = colorError[i] * (d1 * prevA);

                // How this weight affects the X-Slope Error
                float gradXEffect = errorGradX[i] * ((d2 * prevA * rawSlopeX) + (d1 * prevGradX[j]));

                // How this weight affects the Y-Slope Error
                float gradYEffect = errorGradY[i] * ((d2 * prevA * rawSlopeY) + (d1 * prevGradY[j]));

            	// Combine all three to get the total gradient for this specific weight
            	float totalWeightGradient = gradColor + gradXEffect + gradYEffect;
				//totalWeightGradient = std::clamp(totalWeightGradient, -999'999.0f, 999'999.0f);
				if (totalWeightGradient > 999'999'999.f)
				{
					__debugbreak();
				}

            	// Accumulate the 3 Errors to pass back to the previous layer
            	nextColorError[j] += colorError[i] * (d1 * oldWeight) + 
									 errorGradX[i] * (d2 * oldWeight * rawSlopeX) + 
									 errorGradY[i] * (d2 * oldWeight * rawSlopeY);
                
            	nextErrorGradX[j] += errorGradX[i] * (d1 * oldWeight);
            	nextErrorGradY[j] += errorGradY[i] * (d1 * oldWeight);

            	// FIX: ACCUMULATE INTO STORED DELTAS
            	m_storedDelta[l].weights[startWeight + j] += totalWeightGradient;
            }
            
        	// FIX: ACCUMULATE BIASES
        	float gradColorBias = colorError[i] * d1;
        	float gradXBias = errorGradX[i] * (d2 * rawSlopeX);
        	float gradYBias = errorGradY[i] * (d2 * rawSlopeY);
            
        	m_storedDelta[l].biases[i] += (gradColorBias + gradXBias + gradYBias);
        }
        
    	// Transfer the calculated errors for the next loop iteration to use
    	colorError = nextColorError;
    	errorGradX = nextErrorGradX;
    	errorGradY = nextErrorGradY;
    }

	// FIX: Tell the network we stored a batch item
	m_numStored++;
}

void Network::RunGradientGuidedEpoch(
	const std::vector<ImageUtils::SpatialData>& inputData, 
	const std::vector<ImageUtils::SpatialData>& targetData,  
	float learningRate)
{
	SpatialDerivativeBuffer spatialBuffers;
	for (size_t i = 0; i < inputData.size(); ++i)
	{
		const auto& input = inputData[i];
		const auto& target = targetData[i];

		PropagateSpatialDerivativesThreadSafe(input.values, input.gradX, input.gradY, spatialBuffers);
		BackPropagateGradientGuided(target, input.values, spatialBuffers.activations, spatialBuffers.preActivations, spatialBuffers, -195386380931);
	}
	// Apply the averaged batch weights!
	ConsumeDelta(learningRate);
}

void Network::PropagateSpatialDerivativesThreadSafe(
	const std::vector<float>& inputActivation,
    const std::vector<float>& inputGradX,
    const std::vector<float>& inputGradY,
    SpatialDerivativeBuffer& threadBuffers) const
{
	// Resize buffers
	if (threadBuffers.activations.size() != m_layers.size())
	{
		threadBuffers.preActivations.resize(m_layers.size()); // NEW
		threadBuffers.activations.resize(m_layers.size());
		threadBuffers.gradientX.resize(m_layers.size());
		threadBuffers.gradientY.resize(m_layers.size());
        
		for (size_t i = 0; i < m_layers.size(); ++i)
		{
			size_t neurons = m_layers[i]->GetNumNeurons();
			threadBuffers.preActivations[i].resize(neurons);  // NEW
			threadBuffers.activations[i].resize(neurons);
			threadBuffers.gradientX[i].resize(neurons);
			threadBuffers.gradientY[i].resize(neurons);
		}
	}

    // Initialize the input layer
    // The derivative of X with respect to X is 1. The derivative of X with respect to Y is 0.
	std::ranges::copy(inputActivation, threadBuffers.activations[0].begin());
	std::ranges::copy(inputActivation, threadBuffers.preActivations[0].begin()); // NEW
	std::ranges::copy(inputGradX, threadBuffers.gradientX[0].begin());
	std::ranges::copy(inputGradY, threadBuffers.gradientY[0].begin());
    //threadBuffers.gradientX[0] = { 1.0f, 0.0f };
    //threadBuffers.gradientY[0] = { 0.0f, 1.0f };

    // Propagate forward
    for (size_t l = 1; l < m_layers.size(); ++l)
    {
        const Layer* layer = m_layers[l].get();
        const std::vector<float>& prevAct   = threadBuffers.activations[l - 1];
        const std::vector<float>& prevGradX = threadBuffers.gradientX[l - 1];
        const std::vector<float>& prevGradY = threadBuffers.gradientY[l - 1];

        std::vector<float>& currentAct   = threadBuffers.activations[l];
        std::vector<float>& currentGradX = threadBuffers.gradientX[l];
        std::vector<float>& currentGradY = threadBuffers.gradientY[l];

        const size_t numNeurons = layer->GetNumNeurons();
        const size_t weightsPerNeuron = prevAct.size();
        const auto& weights = layer->GetParams().weights;
        const auto& biases = layer->GetParams().biases;
        ActFunc::Base* func = layer->GetActivationFunction();

        for (size_t i = 0; i < numNeurons; ++i)
        {
            float z = biases[i];
            float gradX_z = 0.0f;
            float gradY_z = 0.0f;
            
            size_t weightStart = i * weightsPerNeuron;
            
            // Multiply weights by previous activations AND previous gradients
            for (size_t j = 0; j < weightsPerNeuron; ++j)
            {
                float w = weights[weightStart + j];
                z       += w * prevAct[j];
                gradX_z += w * prevGradX[j];
                gradY_z += w * prevGradY[j];
            }

        	// Save the raw 'z' sum BEFORE applying the activation function
        	threadBuffers.preActivations[l][i] = z;
        	
            // Calculate standard activation
            currentAct[i] = func ? func->Execute(z) : z;
            
            // Calculate the derivative of the activation function at Z
            float actDeriv = func ? func->ExecuteDerivative(z) : 1.0f;
            
            // Chain rule: Multiply the sum of weighted gradients by the activation derivative
            currentGradX[i] = gradX_z * actDeriv;
            currentGradY[i] = gradY_z * actDeriv;
        }
    }
}
