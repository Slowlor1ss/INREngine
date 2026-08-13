#include "GpuNetwork.h"
#include "cublas_utils.h"
#include <iostream>

#include "AdamOptimizer.cuh"
#include "BackwardPass.cuh"
#include "ForwardPass.cuh"

// Main wrapper: Executes one full batch entirely on the GPU
void GpuNetwork::TrainBatchGPU(
    const engineFloat* d_batchInputAct,
    const engineFloat* d_batchInputGradX,
    const engineFloat* d_batchInputGradY,
    const engineFloat* d_batchTargetAct,
    const engineFloat* d_batchTargetGradX,
    const engineFloat* d_batchTargetGradY,
    engineFloat spatialLossWeight,
    engineFloat learningRate)
{
#ifndef _TRAINING
    {
        static int batchCounter = 0;
        cudaError_t ghostErr = cudaGetLastError();
        if ( ghostErr != cudaSuccess ) {
            printf( "\n[GHOST CAUGHT] Error arrived before Batch %d started: %s\n", batchCounter, cudaGetErrorString( ghostErr ) );
            __debugbreak();
        }
        batchCounter++;
    }
#endif
    
    // Update the dynamic LR buffer on the stream (not captured by the graph if done outside)
    CUDA_CHECK(cudaMemcpyAsync(d_learningRate, &learningRate, sizeof(engineFloat), cudaMemcpyHostToDevice, m_stream));

    if (!m_graphCaptured)
    {
        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal));
        
        PROFILE_PUSH_COLOR("ForwardPass", 0xFFFFB3BA); 
        // Pushes the batch of pixels through the network to calculate colors and spatial slopes.
        ForwardPass(d_batchInputAct, d_batchInputGradX, d_batchInputGradY);
        PROFILE_POP();
        
        PROFILE_PUSH_COLOR("BackwardPass", 0xFFBAFFC9); 
        // Calculates the error against the target image and uses cublas to accumulate 
        // the gradients into the d_delta buffers.
        BackwardPass(d_batchTargetAct, d_batchTargetGradX, d_batchTargetGradY, spatialLossWeight);
        PROFILE_POP();

        PROFILE_PUSH_COLOR("ApplyGradientsGPU", 0xFFBAE1FF); 
        // Apply gradients (ADAM OPTIMIZER)
        // Updates all weights, momentum, and velocity in VRAM (auto-clears the deltas to 0.0f)
        ApplyGradientsGPU(learningRate);
        PROFILE_POP();
        
        CUDA_CHECK(cudaStreamEndCapture(m_stream, &m_graph));
        CUDA_CHECK(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));

        m_graphCaptured = true;
    }

    // Launch captured graph
    CUDA_CHECK(cudaGraphLaunch(m_graphExec, m_stream));
}

// Initialize cublas and mirror the CPU network structure to VRAM
GpuNetwork::GpuNetwork(const Network& cpuNetwork, int batchSize)
    : m_batchSize(batchSize)
{
    // Initialize the cublas Hardware Context
    CUBLAS_CHECK(cublasCreate(&m_cublasHandle));
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUBLAS_CHECK(cublasSetStream(m_cublasHandle, m_stream));
    
    m_costType = cpuNetwork.GetCostFunction()->GetGpuType();

    // Mirror every layer to the GPU
    const auto& cpuLayers = cpuNetwork.GetLayers(); 

    // Use an index-based loop so we can peek at layer [i - 1]
    for (size_t i = 0; i < cpuLayers.size(); ++i)
    {
        const auto* cpuLayer = cpuLayers[i].get();
        GpuLayer gpuLayer;

        gpuLayer.numNeurons = static_cast<int>(cpuLayer->GetNumNeurons());
        
        // TODO: this explicit split is probably not needed num neurons should be 0 when using getnumneurons but check to be sure
        if (i == 0) {
            gpuLayer.prevNeurons = 0; // Layer 0 has no previous layer
        } else {
            gpuLayer.prevNeurons = static_cast<int>(cpuLayers[i - 1]->GetNumNeurons());
        }
        
        if (cpuLayer->GetActivationFunction()) {
            gpuLayer.actType = cpuLayer->GetActivationFunction()->GetGpuType();
        } else {
            gpuLayer.actType = GpuActType::None;
        }
        
        gpuLayer.learningRateMultiplier = cpuLayer->GetLearningrateMultiplier();

        const auto& params = cpuLayer->GetParams();
        gpuLayer.hasDualWeights = !params.weights_scale.empty();

        // Allocate the VRAM for this layer
        AllocateLayerMemory(gpuLayer, m_batchSize);
        
        const size_t bSize = gpuLayer.numNeurons * sizeof(engineFloat);
        
        // Verify Biases
#ifndef _TRAINING
        if (params.biases.size() != static_cast<size_t>(gpuLayer.numNeurons)) {
            std::cout << "[FATAL GPU ERROR] Bias dimension mismatch!\n"
                      << "Layer Expected: " << gpuLayer.numNeurons << " biases, but CPU has: " << params.biases.size() << "\n";
            __debugbreak(); 
        }
#endif
        CUDA_CHECK(cudaMemcpy(gpuLayer.d_biases, params.biases.data(), bSize, cudaMemcpyHostToDevice));
        
        if (gpuLayer.hasDualWeights) {
#ifndef _TRAINING
            if (params.biases_scale.size() != static_cast<size_t>(gpuLayer.numNeurons)) {
                std::cout << "[FATAL GPU ERROR] Bias Scale dimension mismatch!\n";
                __debugbreak();
            }
#endif
            CUDA_CHECK(cudaMemcpy(gpuLayer.d_biasesScale, params.biases_scale.data(), bSize, cudaMemcpyHostToDevice));
        }

        // Verify Weights
        if (gpuLayer.prevNeurons > 0) 
        {
            const size_t expectedWeights = static_cast<size_t>(gpuLayer.numNeurons) * gpuLayer.prevNeurons;
            const size_t wSize = expectedWeights * sizeof(engineFloat);
#ifndef _TRAINING
            if (params.weights.size() != expectedWeights) {
                std::cout << "[FATAL GPU ERROR] Weight dimension mismatch!\n"
                          << "Layer Expected: " << expectedWeights << " weights, but CPU has: " << params.weights.size() << "\n";
                __debugbreak();
            }
#endif
            CUDA_CHECK(cudaMemcpy(gpuLayer.d_weights, params.weights.data(), wSize, cudaMemcpyHostToDevice));
            
            if (gpuLayer.hasDualWeights) {
#ifndef _TRAINING
                if (params.weights_scale.size() != expectedWeights) {
                    std::cout << "[FATAL GPU ERROR] Weight Scale dimension mismatch!\n";
                    __debugbreak();
                }
#endif
                CUDA_CHECK(cudaMemcpy(gpuLayer.d_weightsScale, params.weights_scale.data(), wSize, cudaMemcpyHostToDevice));
            }
        }

        m_layers.push_back(gpuLayer);
    }
}

// Clean up all VRAM to prevent memory leaks
GpuNetwork::~GpuNetwork()
{
    for (auto& layer : m_layers) {
        FreeLayerMemory(layer);
    }
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
    cudaStreamDestroy(m_stream);
    cublasDestroy(m_cublasHandle);
}

// Chckpointing Download trained weights from VRAM back to CPU RAM
void GpuNetwork::DownloadParametersToCPU(Network& cpuNetwork)
{
    PROFILE_SCOPE("DownloadParametersToCPU");
    const auto& cpuLayers = cpuNetwork.GetLayers(); 

    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& gpuLayer = m_layers[i];
        auto* cpuLayer = cpuLayers[i].get();
        Parameters& cpuParams = cpuLayer->GetParams(); 

        size_t wSize = gpuLayer.numNeurons * gpuLayer.prevNeurons * sizeof(engineFloat);
        size_t bSize = gpuLayer.numNeurons * sizeof(engineFloat);

        CUDA_CHECK(cudaMemcpy((void**)cpuParams.weights.data(), gpuLayer.d_weights, wSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy((void**)cpuParams.biases.data(), gpuLayer.d_biases, bSize, cudaMemcpyDeviceToHost));

        if (gpuLayer.hasDualWeights) {
            CUDA_CHECK(cudaMemcpy((void**)cpuParams.weights_scale.data(), gpuLayer.d_weightsScale, wSize, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy((void**)cpuParams.biases_scale.data(), gpuLayer.d_biasesScale, bSize, cudaMemcpyDeviceToHost));
        }
    }
}

// Forward pass returning predicted colors to the CPU for rendering
std::vector<engineFloat> GpuNetwork::PredictGPU(
    const engineFloat* d_inputAct, 
    const engineFloat* d_inputGradX, 
    const engineFloat* d_inputGradY)
{
    PROFILE_SCOPE("PredictGPU");
    // Run the forward pass on the GPU
    ForwardPass(d_inputAct, d_inputGradX, d_inputGradY);

    // Grab the final layer
    GpuLayer& outputLayer = m_layers.back();
    size_t outputBytes = m_batchSize * outputLayer.numNeurons * sizeof(engineFloat);

    // TODO: probably not needed as cudaMemcpy blocks
    //CUDA_CHECK(cudaDeviceSynchronize());
#ifndef _TRAINING
	cudaError_t launchErr = cudaPeekAtLastError();
	if ( launchErr != cudaSuccess ) {
		printf( "\n[cudaPeekAtLastError] PredictGPU failed: %s\n", cudaGetErrorString( launchErr ) );
		__debugbreak();
	}
#endif

    // Allocate a CPU vector and copy the results back
    std::vector<engineFloat> predictions(m_batchSize * outputLayer.numNeurons);
    CUDA_CHECK(cudaMemcpy(predictions.data(), outputLayer.d_activations, outputBytes, cudaMemcpyDeviceToHost));
    
    return predictions;
}

// Request raw memory from the GPU
void GpuNetwork::AllocateLayerMemory(GpuLayer& layer, int batchSize)
{
    PROFILE_SCOPE_FMT("AllocateLayerMemory - layer size %d", layer.numNeurons);
    const size_t batchNeuronBytes = batchSize * layer.numNeurons * sizeof(engineFloat);
    const size_t biasBytes = layer.numNeurons * sizeof(engineFloat);
    const size_t weightBytes = layer.numNeurons * layer.prevNeurons * sizeof(engineFloat);

    // Persistent Parameters & Deltas (Biases)
    CUDA_CHECK(cudaMalloc(&layer.d_biases, biasBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_deltaBiases, biasBytes));
    CUDA_CHECK(cudaMemset(layer.d_deltaBiases, 0, biasBytes)); // Must start at 0 for accumulation!

    if (layer.hasDualWeights) {
        CUDA_CHECK(cudaMalloc(&layer.d_biasesScale, biasBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaBiasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_deltaBiasesScale, 0, biasBytes));
    }
    
    // Allocate Adam States (Biases)
    CUDA_CHECK(cudaMalloc(&layer.d_m_biases, biasBytes));
    CUDA_CHECK(cudaMemset(layer.d_m_biases, 0, biasBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_v_biases, biasBytes));
    CUDA_CHECK(cudaMemset(layer.d_v_biases, 0, biasBytes));

    if (layer.hasDualWeights) {
        CUDA_CHECK(cudaMalloc(&layer.d_biasesScale, biasBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaBiasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_deltaBiasesScale, 0, biasBytes));
        
        // Adam States (Biases Scale)
        CUDA_CHECK(cudaMalloc(&layer.d_m_biasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_m_biasesScale, 0, biasBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_v_biasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_v_biasesScale, 0, biasBytes));
    }
    
    if (layer.prevNeurons > 0) {
        CUDA_CHECK(cudaMalloc(&layer.d_weights, weightBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaWeights, weightBytes));
        CUDA_CHECK(cudaMemset(layer.d_deltaWeights, 0, weightBytes)); 

        // Adam States (Weights Freq)
        CUDA_CHECK(cudaMalloc(&layer.d_m_weights, weightBytes));
        CUDA_CHECK(cudaMemset(layer.d_m_weights, 0, weightBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_v_weights, weightBytes));
        CUDA_CHECK(cudaMemset(layer.d_v_weights, 0, weightBytes));

        if (layer.hasDualWeights) {
            CUDA_CHECK(cudaMalloc(&layer.d_weightsScale, weightBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_deltaWeightsScale, weightBytes));
            CUDA_CHECK(cudaMemset(layer.d_deltaWeightsScale, 0, weightBytes));
            
            // Adam States (Weights Scale)
            CUDA_CHECK(cudaMalloc(&layer.d_m_weightsScale, weightBytes));
            CUDA_CHECK(cudaMemset(layer.d_m_weightsScale, 0, weightBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_v_weightsScale, weightBytes));
            CUDA_CHECK(cudaMemset(layer.d_v_weightsScale, 0, weightBytes));
        }
    }

    // Forward Pass Buffers
    CUDA_CHECK(cudaMalloc(&layer.d_preActFreq, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_activations, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_gradX, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_gradY, batchNeuronBytes));

    if (layer.hasDualWeights) {
	    CUDA_CHECK(cudaMalloc(&layer.d_preActScale, batchNeuronBytes));
	    CUDA_CHECK(cudaMalloc(&layer.d_rawSlopeXScale, batchNeuronBytes));
	    CUDA_CHECK(cudaMalloc(&layer.d_rawSlopeYScale, batchNeuronBytes));
	}

    // Backward Pass Buffers (Factored Terms)
    if (layer.prevNeurons > 0) {
        CUDA_CHECK(cudaMalloc(&layer.d_deltaAFreq, batchNeuronBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaXFreq, batchNeuronBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaYFreq, batchNeuronBytes));

        if (layer.hasDualWeights) {
            CUDA_CHECK(cudaMalloc(&layer.d_deltaAScale, batchNeuronBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_deltaXScale, batchNeuronBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_deltaYScale, batchNeuronBytes));
        }
    }

    // Backward Pass Buffers (Incoming Errors)
    CUDA_CHECK(cudaMalloc(&layer.d_colorError, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_errorGradX, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_errorGradY, batchNeuronBytes));
}

// Return memory to the GPU
void GpuNetwork::FreeLayerMemory(GpuLayer& layer)
{
    PROFILE_SCOPE_FMT("FreeLayerMemory - layer size %d", layer.numNeurons);
    auto SafeFree = [](engineFloat*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };

    // Parameters & Deltas
    SafeFree(layer.d_weights);
    SafeFree(layer.d_weightsScale);
    SafeFree(layer.d_biases);
    SafeFree(layer.d_biasesScale);

    SafeFree(layer.d_deltaWeights);
    SafeFree(layer.d_deltaWeightsScale);
    SafeFree(layer.d_deltaBiases);
    SafeFree(layer.d_deltaBiasesScale);

    // Adam States
    SafeFree(layer.d_m_weights);
    SafeFree(layer.d_v_weights);
    SafeFree(layer.d_m_biases);
    SafeFree(layer.d_v_biases);

    SafeFree(layer.d_m_weightsScale);
    SafeFree(layer.d_v_weightsScale);
    SafeFree(layer.d_m_biasesScale);
    SafeFree(layer.d_v_biasesScale);

    // Forward Buffers
    SafeFree(layer.d_preActFreq);
    SafeFree(layer.d_preActScale);
    SafeFree(layer.d_activations);
    SafeFree(layer.d_gradX);
    SafeFree(layer.d_gradY);
    SafeFree(layer.d_rawSlopeXScale);
	SafeFree(layer.d_rawSlopeYScale);

    // Backward Buffers
    SafeFree(layer.d_deltaAFreq);
    SafeFree(layer.d_deltaXFreq);
    SafeFree(layer.d_deltaYFreq);
    SafeFree(layer.d_deltaAScale);
    SafeFree(layer.d_deltaXScale);
    SafeFree(layer.d_deltaYScale);

    // Error Buffers
    SafeFree(layer.d_colorError);
    SafeFree(layer.d_errorGradX);
    SafeFree(layer.d_errorGradY);
}

void GpuNetwork::ForwardPass(const engineFloat* d_batchInputAct, const engineFloat* d_batchInputGradX, const engineFloat* d_batchInputGradY)
{
    // Layer 0 (Input Passthrough)
    GpuLayer& inputLayer = m_layers[0];
    size_t batchNeuronBytes = m_batchSize * inputLayer.numNeurons * sizeof(engineFloat);

    // Copy the raw batch inputs directly into Layer 0's forward buffers
    CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_activations, d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActFreq,  d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_gradX,       d_batchInputGradX, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_gradY,       d_batchInputGradY, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));

    if (inputLayer.hasDualWeights) {
        CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActScale, d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    }
 
    // Forward propagation loop
    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& curr = m_layers[i];
        GpuLayer& prev = m_layers[i - 1];
        
        PROFILE_PUSH_FMT("Forward propagation loop - layer size %d", curr.numNeurons);
        // Launch the Forward Kernel for this specific layer
        RunForwardLayerGPU(
            m_cublasHandle, m_stream,
            m_batchSize, curr.numNeurons, prev.numNeurons,
            curr.d_weights, curr.d_weightsScale, curr.d_biases, curr.d_biasesScale,
            prev.d_activations, prev.d_gradX, prev.d_gradY,
            curr.d_preActFreq, curr.d_preActScale,
            curr.d_activations, curr.d_gradX, curr.d_gradY,
            curr.d_rawSlopeXScale, curr.d_rawSlopeYScale,
            curr.actType, curr.hasDualWeights
        );
        PROFILE_POP();
    }
}

void GpuNetwork::BackwardPass(
    const engineFloat* d_batchTargetAct, 
    const engineFloat* d_batchTargetGradX, 
    const engineFloat* d_batchTargetGradY, 
    engineFloat spatialLossWeight)
{
    // Calculate output error
    GpuLayer& outputLayer = m_layers.back();
    
    CalculateOutputErrorGPU(
        m_stream,
        m_batchSize, outputLayer.numNeurons,
        outputLayer.d_activations, outputLayer.d_gradX, outputLayer.d_gradY,
        d_batchTargetAct, d_batchTargetGradX, d_batchTargetGradY,
        outputLayer.d_colorError, outputLayer.d_errorGradX, outputLayer.d_errorGradY,
        spatialLossWeight, m_costType
    );
    
    // Update All Layers (Looping from Output down to Layer 0)
    for (int i = (int)m_layers.size() - 1; i > 0; --i)
    {
        GpuLayer& curr = m_layers[i];
        GpuLayer& prev = m_layers[i - 1];

        PROFILE_PUSH_FMT("Backward propagation loop - layer size %d", curr.numNeurons);
        // We must wipe the previous layer's error buffers to zero before cublas accumulates into them
        size_t batchNeuronBytes = m_batchSize * prev.numNeurons * sizeof(engineFloat);
        CUDA_CHECK(cudaMemsetAsync(prev.d_colorError,  0, batchNeuronBytes, m_stream));
        CUDA_CHECK(cudaMemsetAsync(prev.d_errorGradX,  0, batchNeuronBytes, m_stream));
        CUDA_CHECK(cudaMemsetAsync(prev.d_errorGradY,  0, batchNeuronBytes, m_stream));

        RunBackwardLayerGPU(
            m_cublasHandle, m_stream,
            m_batchSize, curr.numNeurons, prev.numNeurons,
            curr.d_colorError, curr.d_errorGradX, curr.d_errorGradY,
            prev.d_activations, prev.d_gradX, prev.d_gradY,
            curr.d_weights, curr.d_weightsScale,
            curr.d_preActFreq, curr.d_preActScale,
            curr.d_deltaAFreq, curr.d_deltaXFreq, curr.d_deltaYFreq,
            curr.d_deltaAScale, curr.d_deltaXScale, curr.d_deltaYScale,
            prev.d_colorError, prev.d_errorGradX, prev.d_errorGradY,
            curr.d_deltaWeights, curr.d_deltaWeightsScale,
            curr.d_deltaBiases, curr.d_deltaBiasesScale,
            curr.actType, curr.hasDualWeights
        );
        PROFILE_POP();
    }
}

void GpuNetwork::ApplyGradientsGPU(engineFloat baseLearningRate)
{
    // Layer 0 has no weights to optimize, start at 1
    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& layer = m_layers[i];
        
        // Advance the time step for this layer
        layer.adam_t += 1;
        
        // Calculate total elements
        int numWeights = layer.numNeurons * layer.prevNeurons;
        int numBiases = layer.numNeurons;

        // Note: layer.learningRateMultiplier should be handled either inside the kernel or 
        // by scaling the d_learningRate per layer if they vary. Assuming it's 1.0 like now, 
        // or you can easily pass it to the kernel to multiply against *d_learningRate.
        //engineFloat actualLR = baseLearningRate * layer.learningRateMultiplier;
        
        // Update Weights
        RunAdamOptimizerGPU(m_stream, numWeights, layer.d_weights, layer.d_deltaWeights,
                            layer.d_m_weights, layer.d_v_weights, d_learningRate, layer.adam_t, m_batchSize);

        // Update Biases
        RunAdamOptimizerGPU(m_stream, numBiases, layer.d_biases, layer.d_deltaBiases,
                            layer.d_m_biases, layer.d_v_biases, d_learningRate, layer.adam_t, m_batchSize);

        // Update Dual Weights (if applicable)
        if (layer.hasDualWeights) {
            RunAdamOptimizerGPU(m_stream, numWeights, layer.d_weightsScale, layer.d_deltaWeightsScale,
                                layer.d_m_weightsScale, layer.d_v_weightsScale, d_learningRate, layer.adam_t, m_batchSize);

            RunAdamOptimizerGPU(m_stream, numBiases, layer.d_biasesScale, layer.d_deltaBiasesScale,
                                layer.d_m_biasesScale, layer.d_v_biasesScale, d_learningRate, layer.adam_t, m_batchSize);
        }
    }
}

// Kept for debugging purposes
// void GpuNetwork::TransferDeltasToCPU(Network& cpuNetwork)
// {
//     // Grab the CPU array where the Adam optimizer expects to find the gradients
//     auto& storedDeltas = cpuNetwork.GetStoredDeltas();
//
//     // We start at 1 because Layer 0 is just the input passthrough (no weights)
//     for (size_t i = 1; i < m_layers.size(); ++i)
//     {
//         GpuLayer& gpuLayer = m_layers[i];
//         Parameters& deltaParams = storedDeltas[i]; 
//         
//         size_t wSize = gpuLayer.numNeurons * gpuLayer.prevNeurons * sizeof(engineFloat);
//         size_t bSize = gpuLayer.numNeurons * sizeof(engineFloat);
//
//         // Copy gradients from Device (VRAM) directly into the CPU's stored delta arrays
//         CUDA_CHECK(cudaMemcpy(deltaParams.weights.data(), gpuLayer.d_deltaWeights, wSize, cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(deltaParams.biases.data(), gpuLayer.d_deltaBiases, bSize, cudaMemcpyDeviceToHost));
//
//         if (gpuLayer.hasDualWeights) {
//             CUDA_CHECK(cudaMemcpy(deltaParams.weights_scale.data(), gpuLayer.d_deltaWeightsScale, wSize, cudaMemcpyDeviceToHost));
//             CUDA_CHECK(cudaMemcpy(deltaParams.biases_scale.data(), gpuLayer.d_deltaBiasesScale, bSize, cudaMemcpyDeviceToHost));
//         }
//     }
// }
