#include "Network.h"
#include "Layer.h"
#include "ActFuncDataBase.h"
#include "iostream"

Network::Network(
const std::vector<size_t>& neuronsPerLayer, 
	const std::vector<ActFunc::Base*>& activations, 
	CostFunc::Base* costFunc, 
	bool zeroInit)
{
	if (activations.size() != neuronsPerLayer.size() - 1)
	{
		throw std::invalid_argument(
			"Network topology mismatch! The number of activation functions "
			"must be exactly one less than the number of layers."
		);
	}
	
    m_costFunction = (costFunc != nullptr) ? costFunc : CostFunc::DataBase::FindCostFunc<CostFunc::L1>();

    m_layers.push_back(std::make_unique<InitialLayer>(neuronsPerLayer[0]));
    m_storedDelta.emplace_back(m_layers.front()->GetNumNeurons(), m_layers.front()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), 0);
	m_storedDelta.back().Clear(); // TODO: unsure
    m_costDeltas.emplace_back(neuronsPerLayer[0], 0.0f);

    for (size_t i = 1; i < neuronsPerLayer.size(); i++)
    {
        // Choose the activation function based on the layer index
        //ActFunc::Base* actFunc = (i == neuronsPerLayer.size() - 1) ? outputActivation : hiddenActivation;
    	ActFunc::Base* actFunc = activations[i - 1];

        m_layers.push_back(std::make_unique<Layer>(neuronsPerLayer[i], actFunc, i, m_layers.back().get()));
        
    	// Fetch the default multiplier for this layer's activation function
    	m_layers.back()->m_learningRateMultiplier = actFunc ? actFunc->GetLearningRateMultiplier() : 1.0f;
    	
    	if (m_layers.size() > 2)
    	{
    		Layer* outputLayer = m_layers.back().get();
    		Layer* lastHiddenLayer = m_layers[m_layers.size() - 2].get();

    		// If the output activation is "None", inherit from the last hidden layer
    		if (outputLayer->m_activationFunction->GetName() == ActFunc::None::k_name)
    		{
    			outputLayer->m_learningRateMultiplier = lastHiddenLayer->m_learningRateMultiplier;
    		}
    	}
    	
    	//m_storedDelta.emplace_back(m_layers.back()->GetNumNeurons(), m_layers.back()->GetNumWeightsToPrevious(), ActFunc::DataBase::FindActFunc<ActFunc::None>(), i);
    	// We pass the actfunc rather then none so we can chek for the stored deltas and by extend the local deltas weathe rthe act function is wire
    	// TODO: maybe a better solution
        m_storedDelta.emplace_back(m_layers.back()->GetNumNeurons(), m_layers.back()->GetNumWeightsToPrevious(), actFunc, i);
		m_storedDelta.back().Clear(); // TODO: unsure
    	m_costDeltas.emplace_back(neuronsPerLayer[i], 0.0f);
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

InitialLayer& Network::GetInitialLayer() const
{
	return *static_cast<InitialLayer*>(m_layers.front().get());
}

// Network.cpp
void Network::ConsumeDelta(engineFloat learningRate)
{
	if (m_layers.size() == m_storedDelta.size() && m_numStored > 0)
	{
		for (size_t i = 1; i < m_layers.size(); i++)
		{
			//const engineFloat lr = learningRate * m_layers[i]->m_activationFunction->GetLearningRateMultiplier();
			const engineFloat lr = learningRate * m_layers[i]->m_learningRateMultiplier;
			
			if (m_layers[i]->m_params.biases.size() == m_storedDelta[i].biases.size()
				&& m_layers[i]->m_params.weights.size() == m_storedDelta[i].weights.size())
			{
				// Average the accumulated gradients over the batch
				m_storedDelta[i] *= (1.0f / static_cast<engineFloat>(m_numStored));

				// Pass the averaged gradients to our new Adam optimizer!
				// (Note: ApplyAdamUpdate handles the learning rate and subtraction internally)
				m_layers[i]->m_params.ApplyAdamUpdate(
					m_storedDelta[i].weights, 
					m_storedDelta[i].biases, 
					m_storedDelta[i].weights_scale, 
					m_storedDelta[i].biases_scale, 
					lr
				);

				// Clear the delta buffer for the next batch
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

// engineFloat Network::CalculateCost(const std::vector<engineFloat>& inputActivation,const std::vector<engineFloat>& preferredOutput)
// {
// 	const std::vector<engineFloat>& result = Propagate(inputActivation);
//
// 	engineFloat cost = 0.0f;
// 	if (result.size() == preferredOutput.size())
// 	{
// 		for (size_t i = 0; i < result.size(); i++)
// 		{
// 			cost += m_costFunction->Execute(result[i], preferredOutput[i]);
// 		}
// 		return cost / result.size();
// 	}
//
// 	return std::numeric_limits<engineFloat>().infinity();
// }

// std::vector<engineFloat> Network::Propagate(const std::vector<engineFloat>& inputActivation)
// {
// 	GetInitialLayer().StartPropagation(inputActivation);
// 	return m_layers.back()->m_activations;
// }

const std::vector<engineFloat>& Network::PropagateThreadSafe(const std::vector<engineFloat>& inputActivation, std::vector<std::vector<engineFloat>>& threadBuffers) const
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
		const std::vector<engineFloat>& prevAct = threadBuffers[l - 1];
		std::vector<engineFloat>& currentAct = threadBuffers[l];

		const size_t numNeurons = layer->GetNumNeurons();
		const size_t weightsPerNeuron = prevAct.size();
		const auto& weights = layer->GetParams().weights;
		const auto& biases = layer->GetParams().biases;
		ActFunc::Base* func = layer->GetActivationFunction();
		
		bool hasDualWeights = !layer->GetParams().weights_scale.empty();
		for (size_t i = 0; i < numNeurons; ++i)
		{
			engineFloat zFreq = biases[i];
			engineFloat zScale = hasDualWeights ? layer->GetParams().biases_scale[i] : 0.0f;
    
			size_t weightStart = i * weightsPerNeuron;
			for (size_t j = 0; j < weightsPerNeuron; ++j)
			{
				// Calculate the frequency sum
				zFreq += weights[weightStart + j] * prevAct[j];
        
				// Calculate the scale sum
				if (hasDualWeights) {
					zScale += layer->GetParams().weights_scale[weightStart + j] * prevAct[j];
				}
			}

			// Execute using the dual Z values!
			currentAct[i] = func ? func->ExecuteDual(zFreq, zScale) : zFreq;
		}
	}

	return threadBuffers.back();
}

// engineFloat Network::BackPropagate(const std::vector<engineFloat>& inputActivation, const std::vector<engineFloat>& preferredOutput)
// {
// 	// propagate forwards
// 	engineFloat cost = CalculateCost(inputActivation, preferredOutput);
//
// 	// create empty network with same dimensions to store deltas. 
// 	// propagate backwards
// 	Layer* layer = m_layers.back().get();
// 	size_t layerIndex = m_layers.size() - 1;
//
// 	std::vector<engineFloat>& prevLayerCostDeltas = m_costDeltas[layerIndex];
// 	for (size_t i = 0; i < layer->m_numNeurons; i++)
// 	{
// 		engineFloat act = layer->m_activations[i];
// 		engineFloat y = preferredOutput[i];
//
// 		// the derivative of the cost function
// 		// a.k.a direction to push our activation to decrease cost ?
// 		prevLayerCostDeltas[i] = m_costFunction->ExecuteDerivative(act, y);
// 	}
//
// 	while (layer->m_previousLayer != nullptr)
// 	{
// 		Parameters& deltaLayer = m_storedDelta[layerIndex];
// 		const std::vector<engineFloat>& currentCostDeltas = m_costDeltas[layerIndex];
//
// 		for (size_t i = 0; i < layer->m_numNeurons; i++)
// 		{
// 			//calc bias nudge
// 			engineFloat z = layer->m_preProcessedActivations[i];
// 			engineFloat actFuncDeriv = layer->m_activationFunction->ExecuteDerivative(z);
//
// 			deltaLayer.biases[i] += actFuncDeriv * currentCostDeltas[i];
//
// 			//calc weight nudge (bias nudge * A(L-1)) for each weight
// 			size_t weightsPerNeuron = layer->m_previousLayer->m_numNeurons;
// 			size_t weightIndexStart = i * weightsPerNeuron;
//
// 			// this because we have a weight for each neuron in the previous layer,
// 			// might not always be the case.
// 			for (size_t j = 0; j < layer->m_previousLayer->m_numNeurons; j++)
// 			{
// 				deltaLayer.weights[weightIndexStart + j] += layer->m_previousLayer->m_activations[j] * actFuncDeriv * currentCostDeltas[i];
// 			}
// 		}
//
// 		// setup for next layer
// 		if (layerIndex > 1)
// 		{
// 			std::vector<engineFloat>& nextCostDeltas = m_costDeltas[layerIndex - 1];
// 			std::fill(nextCostDeltas.begin(), nextCostDeltas.end(), 0.0f);
//
// 			for (size_t i = 0; i < layer->m_previousLayer->m_numNeurons; i++)
// 			{
// 				// for each outgoing weight from neuron[j]
// 				// calc weight * weightDestNeuron.biasDelta;
// 				// sum it. save in prevLayerCostDeltas[j]
// 				size_t relevantWeightIndex = i;
// 				size_t weightsPerNeuron = layer->m_previousLayer->m_numNeurons;
// 				for (size_t j = 0; j < layer->m_numNeurons; j++)
// 				{
// 					size_t weightIdx = weightsPerNeuron * j + relevantWeightIndex;
// 					engineFloat z = layer->m_preProcessedActivations[j];
// 					engineFloat actFuncDeriv = layer->m_activationFunction->ExecuteDerivative(z);
// 					engineFloat biasDelta = actFuncDeriv * currentCostDeltas[j];
// 					nextCostDeltas[i] += layer->m_params.weights[weightIdx] * biasDelta; // deltaLayer.biases[j] == actFuncDeriv * currentCostDeltas[i];
// 				}
// 			}
// 		}
//
// 		layer = layer->m_previousLayer;
// 		layerIndex--;
// 	}
//
// 	// update weights and biases
// 	m_numStored++;
//
// 	return cost;
// }

void Network::AccumulateWorkerDeltas(const std::vector<Parameters>& workerDeltas, size_t workerNumStored)
{
    if (m_storedDelta.size() == workerDeltas.size())
    {
        for (size_t i = 1; i < m_storedDelta.size(); i++)
        {
            m_storedDelta[i] += workerDeltas[i];
        }
        m_numStored += workerNumStored;
    }
}

void Network::BackPropagateGradientGuided(
	    const ImageUtils::SpatialData& target,
	    const std::vector<engineFloat>& input,
	    const std::vector<std::vector<engineFloat>>& forwardActivations,
	    const std::vector<std::vector<engineFloat>>& forwardSums,
	    const SpatialDerivativeBuffer& spatialBuffers,   // Network's predicted slopes
	    engineFloat learningRate,
	    std::vector<Parameters>& localDeltas,
	    size_t& localNumStored
	)
{
    // ------
    // Calculate the 3 Error Signals at the Output Layer
    // ------
    const size_t numLayers = m_layers.size();
    const auto& outputActivations = forwardActivations.back();
    const auto& outputGradX = spatialBuffers.gradientX.back();
    const auto& outputGradY = spatialBuffers.gradientY.back();

    std::vector<engineFloat> colorError(outputActivations.size());
    std::vector<engineFloat> errorGradX(outputActivations.size());
    std::vector<engineFloat> errorGradY(outputActivations.size());

    //for (size_t i = 0; i < outputActivations.size(); ++i) {
    //    // Derivative of (Prediction - Target)^2 is 2 * (Prediction - Target)
    //    colorError[i] = 2.0f * (outputActivations[i] - target.values[i]);
    //    errorGradX[i] = 2.0f * (outputGradX[i] - target.gradX[i]);
    //    errorGradY[i] = 2.0f * (outputGradY[i] - target.gradY[i]);
    //}

	// A hyperparameter to balance how much the network cares about slopes vs colors.
    // 0.01f is a great starting point so the massive slopes don't nuke the colors.
    constexpr engineFloat spatialLossWeight = 0.f;//0.0001f;//0.00001f; //TODO: RENABLE

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
        // const std::vector<EngineFloat>& prevActivations = (l == 0) ? input : forwardActivations[l - 1]; 
        // // Grab the previous layer's slopes (or the target's input slopes if layer 0)
        // const std::vector<EngineFloat>& prevGradX = (l == 0) ? target.gradX : spatialBuffers.gradientX[l - 1]; 
        // const std::vector<EngineFloat>& prevGradY = (l == 0) ? target.gradY : spatialBuffers.gradientY[l - 1]; 
    	
    	// Grab the previous layer's data directly from the buffers
    	const std::vector<engineFloat>& prevActivations = forwardActivations[l - 1]; 
    	const std::vector<engineFloat>& prevGradX = spatialBuffers.gradientX[l - 1]; 
    	const std::vector<engineFloat>& prevGradY = spatialBuffers.gradientY[l - 1];
    	
        size_t weightsPerNeuron = prevActivations.size();
        
        // Prepare empty error arrays to pass back to the previous layer
        std::vector<engineFloat> nextColorError(weightsPerNeuron, 0.0f);
        std::vector<engineFloat> nextErrorGradX(weightsPerNeuron, 0.0f);
        std::vector<engineFloat> nextErrorGradY(weightsPerNeuron, 0.0f);
    	
    	bool hasDualWeights = !m_layers[l]->m_params.weights_scale.empty();
    	auto* actFunc = m_layers[l]->m_activationFunction;
        for (size_t i = 0; i < m_layers[l]->GetNumNeurons(); ++i) 
        {
            const engineFloat zFreq = forwardSums[l][i];
        	const engineFloat zScale = spatialBuffers.preActivationsScale[l][i];
        	
        	const auto [deriv1Freq, deriv1Scale] = actFunc->ExecuteDualDerivative(zFreq, zScale); // f'(z)
        	const auto [deriv2Freq, deriv2Scale, deriv2Mixed] = actFunc->ExecuteDualSecondDerivative(zFreq, zScale); // f''(z)

            const size_t startWeight = i * weightsPerNeuron;
            
            // To calculate f''(z) * Sum(w * prev_slope), we need the raw pre-activation slope
            engineFloat rawSlopeXFreq = 0.0f;
            engineFloat rawSlopeXScale = 0.0f;
            engineFloat rawSlopeYFreq = 0.0f;
            engineFloat rawSlopeYScale = 0.0f;
        	
            for (size_t j = 0; j < prevActivations.size(); ++j) {
                const engineFloat weight = m_layers[l]->m_params.weights[startWeight + j];
                rawSlopeXFreq += weight * prevGradX[j];
                rawSlopeYFreq += weight * prevGradY[j];
            	
            	if (hasDualWeights) {
            		const engineFloat weightScale = m_layers[l]->m_params.weights_scale[startWeight + j];
            		rawSlopeXScale += weightScale * prevGradX[j];
            		rawSlopeYScale += weightScale * prevGradY[j];
            	}
            }

            for (size_t j = 0; j < prevActivations.size(); ++j) 
            {
                const engineFloat prevA = prevActivations[j];
            	
            	// Freq gradients
                const engineFloat oldWeightFreq = m_layers[l]->m_params.weights[startWeight + j]; // Save before modifying

                // How this weight affects the Color Error
                const engineFloat gradColorFreq = colorError[i] * (deriv1Freq * prevA);
                // How this weight affects the X-Slope Error
            	const engineFloat gradXEffectFreq = errorGradX[i] * ((deriv2Freq * prevA * rawSlopeXFreq) + (deriv2Mixed * prevA * rawSlopeXScale) + (deriv1Freq * prevGradX[j]));
            	// How this weight affects the Y-Slope Error
            	const engineFloat gradYEffectFreq = errorGradY[i] * ((deriv2Freq * prevA * rawSlopeYFreq) + (deriv2Mixed * prevA * rawSlopeYScale) + (deriv1Freq * prevGradY[j]));

            	// Combine all three to get the total gradient for this specific weight
            	engineFloat totalWeightGradientFreq = gradColorFreq + gradXEffectFreq + gradYEffectFreq;
				//totalWeightGradient = std::clamp(totalWeightGradient, -999'999.0f, 999'999.0f);
#ifndef _TRAINING
            	if (totalWeightGradientFreq > 999'999'999.f)
				{
					totalWeightGradientFreq = 999'999.f;
					__debugbreak();
				}
#endif
            	localDeltas[l].weights[startWeight + j] += totalWeightGradientFreq;
            	
            	// Scale gradients
            	engineFloat oldWeightScale = 0.0f;
            	if (hasDualWeights) {
            		oldWeightScale = m_layers[l]->m_params.weights_scale[startWeight + j];
            
            		const engineFloat gradColorScale = colorError[i] * (deriv1Scale * prevA);
            		const engineFloat gradXEffectScale = errorGradX[i] * ((deriv2Mixed * prevA * rawSlopeXFreq) + (deriv2Scale * prevA * rawSlopeXScale) + (deriv1Scale * prevGradX[j]));
            		const engineFloat gradYEffectScale = errorGradY[i] * ((deriv2Mixed * prevA * rawSlopeYFreq) + (deriv2Scale * prevA * rawSlopeYScale) + (deriv1Scale * prevGradY[j]));
            		
            		engineFloat totalWeightGradientScale = gradColorScale + gradXEffectScale + gradYEffectScale;
#ifndef _TRAINING
            		if (totalWeightGradientScale > 999'999'999.f)
            		{
            			totalWeightGradientScale = 999'999.f;
            			__debugbreak();
            		}
#endif
            		localDeltas[l].weights_scale[startWeight + j] += totalWeightGradientScale;
            	}

            	// Accumulate errors to pass back to the previous layer
            	nextColorError[j] += colorError[i] * ((deriv1Freq * oldWeightFreq) + (deriv1Scale * oldWeightScale)) + 
									 errorGradX[i] * ((deriv2Freq * oldWeightFreq * rawSlopeXFreq) + (deriv2Scale * oldWeightScale * rawSlopeXScale)) + 
									 errorGradY[i] * ((deriv2Freq * oldWeightFreq * rawSlopeYFreq) + (deriv2Scale * oldWeightScale * rawSlopeYScale));
        
            	nextErrorGradX[j] += errorGradX[i] * ((deriv1Freq * oldWeightFreq) + (deriv1Scale * oldWeightScale));
            	nextErrorGradY[j] += errorGradY[i] * ((deriv1Freq * oldWeightFreq) + (deriv1Scale * oldWeightScale));
            }
            
        	// Accumulate biases
        	const engineFloat gradColorBiasFreq = colorError[i] * deriv1Freq;
        	const engineFloat gradXBiasFreq = errorGradX[i] * ((deriv2Freq * rawSlopeXFreq) + (deriv2Mixed * rawSlopeXScale));
        	const engineFloat gradYBiasFreq = errorGradY[i] * ((deriv2Freq * rawSlopeYFreq) + (deriv2Mixed * rawSlopeYScale));
        	localDeltas[l].biases[i] += (gradColorBiasFreq + gradXBiasFreq + gradYBiasFreq);
        	
        	if (hasDualWeights) {
        		const engineFloat gradColorBiasScale = colorError[i] * deriv1Scale;
        		const engineFloat gradXBiasScale = errorGradX[i] * ((deriv2Scale * rawSlopeXScale) + (deriv2Mixed * rawSlopeXFreq));
        		const engineFloat gradYBiasScale = errorGradY[i] * ((deriv2Scale * rawSlopeYScale) + (deriv2Mixed * rawSlopeYFreq));
        		localDeltas[l].biases_scale[i] += (gradColorBiasScale + gradXBiasScale + gradYBiasScale);
        	}
        }
        
    	// Transfer the calculated errors for the next loop iteration to use
    	colorError = nextColorError;
    	errorGradX = nextErrorGradX;
    	errorGradY = nextErrorGradY;
    }

	localNumStored++;
}

// void Network::RunGradientGuidedEpoch(
// 	const std::vector<ImageUtils::SpatialData>& inputData, 
// 	const std::vector<ImageUtils::SpatialData>& targetData,
// 	const engineFloat learningRate)
// {
// 	SpatialDerivativeBuffer spatialBuffers;
// 	for (size_t i = 0; i < inputData.size(); ++i)
// 	{
// 		const auto& input = inputData[i];
// 		const auto& target = targetData[i];
//
// 		PropagateSpatialDerivativesThreadSafe(input.values, input.gradX, input.gradY, spatialBuffers);
// 		BackPropagateGradientGuided(target, input.values, spatialBuffers.activations, spatialBuffers.preActivations,
// 		                            spatialBuffers, -195386380931, m_storedDelta, m_numStored); // TODO: remove learning rate
// 	}
// 	// Apply the averaged batch weights!
// 	ConsumeDelta(learningRate);
// }

void Network::PropagateSpatialDerivativesThreadSafe(
	const std::vector<engineFloat>& inputActivation,
    const std::vector<engineFloat>& inputGradX,
    const std::vector<engineFloat>& inputGradY,
    SpatialDerivativeBuffer& threadBuffers) const
{
	// Resize buffers
	if (threadBuffers.activations.size() != m_layers.size())
	{
		threadBuffers.preActivations.resize(m_layers.size());
		threadBuffers.preActivationsScale.resize(m_layers.size());
		threadBuffers.activations.resize(m_layers.size());
		threadBuffers.gradientX.resize(m_layers.size());
		threadBuffers.gradientY.resize(m_layers.size());
        
		for (size_t i = 0; i < m_layers.size(); ++i)
		{
			size_t neurons = m_layers[i]->GetNumNeurons();
			threadBuffers.preActivations[i].resize(neurons);
			threadBuffers.preActivationsScale[i].resize(neurons);
			threadBuffers.activations[i].resize(neurons);
			threadBuffers.gradientX[i].resize(neurons);
			threadBuffers.gradientY[i].resize(neurons);
		}
	}

    // Initialize the input layer
    // The derivative of X with respect to X is 1. The derivative of X with respect to Y is 0.
	std::ranges::copy(inputActivation, threadBuffers.activations[0].begin());
	std::ranges::copy(inputActivation, threadBuffers.preActivations[0].begin());
	std::ranges::copy(inputActivation, threadBuffers.preActivationsScale[0].begin());
	std::ranges::copy(inputGradX, threadBuffers.gradientX[0].begin());
	std::ranges::copy(inputGradY, threadBuffers.gradientY[0].begin());
    //threadBuffers.gradientX[0] = { 1.0f, 0.0f };
    //threadBuffers.gradientY[0] = { 0.0f, 1.0f };

    // Propagate forward
    for (size_t l = 1; l < m_layers.size(); ++l)
    {
        const Layer* layer = m_layers[l].get();
        const std::vector<engineFloat>& prevAct   = threadBuffers.activations[l - 1];
        const std::vector<engineFloat>& prevGradX = threadBuffers.gradientX[l - 1];
        const std::vector<engineFloat>& prevGradY = threadBuffers.gradientY[l - 1];

        std::vector<engineFloat>& currentAct   = threadBuffers.activations[l];
        std::vector<engineFloat>& currentGradX = threadBuffers.gradientX[l];
        std::vector<engineFloat>& currentGradY = threadBuffers.gradientY[l];

        const size_t numNeurons = layer->GetNumNeurons();
        const size_t weightsPerNeuron = prevAct.size();
        const auto& weights = layer->GetParams().weights;
        const auto& biases = layer->GetParams().biases;
        const ActFunc::Base* func = layer->GetActivationFunction();
		
    	bool hasDualWeights = !layer->GetParams().weights_scale.empty();
        for (size_t i = 0; i < numNeurons; ++i)
        {
            engineFloat zFreq = biases[i];
        	engineFloat zScale = hasDualWeights ? layer->GetParams().biases_scale[i] : 0.0f;
        	
        	engineFloat gradXZFreq = 0.0f; 
        	engineFloat gradXZScale = 0.0f;
        	engineFloat gradYZFreq = 0.0f; 
        	engineFloat gradYZScale = 0.0f;

            const size_t weightStart = i * weightsPerNeuron;
            
            // Multiply weights by previous activations AND previous gradients
            for (size_t j = 0; j < weightsPerNeuron; ++j)
            {
            	// Frequency Path (Standard)
            	const engineFloat wF = weights[weightStart + j];
            	zFreq += wF * prevAct[j];
            	gradXZFreq += wF * prevGradX[j];
            	gradYZFreq += wF * prevGradY[j];

            	// Scale Path (Dual Weights)
            	if (hasDualWeights) {
            		const engineFloat wS = layer->GetParams().weights_scale[weightStart + j];
            		zScale += wS * prevAct[j];
            		gradXZScale += wS * prevGradX[j];
            		gradYZScale += wS * prevGradY[j];
            	}
            }

        	// Save the raw 'z' sum BEFORE applying the activation function
        	threadBuffers.preActivations[l][i] = zFreq;
        	threadBuffers.preActivationsScale[l][i] = zScale;
        	
            // Calculate standard activation
        	currentAct[i] = func ? func->ExecuteDual(zFreq, zScale) : zFreq;
            
            // Calculate the derivative of the activation function at Z
            //const engineFloat actDeriv = func ? func->ExecuteDerivative(zFreq) : 1.0f;
        	const auto [deriv1Freq, deriv1Scale] = func ? func->ExecuteDualDerivative(zFreq, zScale) : std::make_pair(1.0f, 0.0f);
            
            // Chain rule: Multiply the sum of weighted gradients by the activation derivative
        	currentGradX[i] = (gradXZFreq * deriv1Freq) + (gradXZScale * deriv1Scale);
        	currentGradY[i] = (gradYZFreq * deriv1Freq) + (gradYZScale * deriv1Scale);
        }
    }
}
