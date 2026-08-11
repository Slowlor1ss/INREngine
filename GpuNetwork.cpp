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
    // Pushes the batch of pixels through the network to calculate colors and spatial slopes.
    ForwardPass(d_batchInputAct, d_batchInputGradX, d_batchInputGradY);
    
    // Calculates the error against the target image and uses cublas to accumulate 
    // the gradients into the d_delta buffers.
    BackwardPass(d_batchTargetAct, d_batchTargetGradX, d_batchTargetGradY, spatialLossWeight);

    // Apply gradients (ADAM OPTIMIZER)
    // Updates all weights, momentum, and velocity in VRAM (auto-clears the deltas to 0.0f)
    ApplyGradientsGPU(learningRate);
}

// Initialize cublas and mirror the CPU network structure to VRAM
GpuNetwork::GpuNetwork(const Network& cpuNetwork, int batchSize)
    : m_batchSize(batchSize)
{
    // Initialize the cuBLAS Hardware Context
    CUBLAS_CHECK(cublasCreate(&m_cublasHandle));
    
    m_costType = cpuNetwork.GetCostFunction()->GetGpuType();

    // Mirror every layer to the GPU
    const auto& cpuLayers = cpuNetwork.GetLayers(); 

    for (const auto& layer : cpuLayers)
    {
        const auto* cpuLayer = layer.get();
        GpuLayer gpuLayer;

        // TODO: make num neurons in Network int as cuda blas requires int's
        gpuLayer.numNeurons = static_cast<int>(cpuLayer->GetNumNeurons());
        gpuLayer.prevNeurons = static_cast<int>(cpuLayer->GetNumWeightsToPrevious());
        
        if (cpuLayer->GetActivationFunction()) {
            gpuLayer.actType = cpuLayer->GetActivationFunction()->GetGpuType();
        } else {
            gpuLayer.actType = GpuActType::None;
        }

        const auto& params = cpuLayer->GetParams();
        gpuLayer.hasDualWeights = !params.weights_scale.empty();

        // Allocate the VRAM for this layer
        AllocateLayerMemory(gpuLayer, m_batchSize);

        // Copy persistent weights/biases from CPU to GPU
        const size_t bSize = gpuLayer.numNeurons * sizeof(engineFloat);
        CUDA_CHECK(cudaMemcpy(gpuLayer.d_biases, params.biases.data(), bSize, cudaMemcpyHostToDevice));
        
        if (gpuLayer.hasDualWeights) {
            CUDA_CHECK(cudaMemcpy(gpuLayer.d_biasesScale, params.biases_scale.data(), bSize, cudaMemcpyHostToDevice));
        }

        if (gpuLayer.prevNeurons > 0) 
        {
            const size_t wSize = static_cast<size_t>(gpuLayer.numNeurons) * gpuLayer.prevNeurons * sizeof(engineFloat);
            CUDA_CHECK(cudaMemcpy(gpuLayer.d_weights, params.weights.data(), wSize, cudaMemcpyHostToDevice));
            
            if (gpuLayer.hasDualWeights) {
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
    cublasDestroy(m_cublasHandle);
}

// Chckpointing Download trained weights from VRAM back to CPU RAM
void GpuNetwork::DownloadParametersToCPU(Network& cpuNetwork)
{
    const auto& cpuLayers = cpuNetwork.GetLayers(); 

    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& gpuLayer = m_layers[i];
        auto* cpuLayer = cpuLayers[i].get();
        auto& cpuParams = cpuLayer->GetParams(); 

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
    // Run the forward pass on the GPU
    ForwardPass(d_inputAct, d_inputGradX, d_inputGradY);

    // Grab the final layer
    GpuLayer& outputLayer = m_layers.back();
    size_t outputBytes = m_batchSize * outputLayer.numNeurons * sizeof(engineFloat);
    
    // Allocate a CPU vector and copy the results back
    std::vector<engineFloat> predictions(m_batchSize * outputLayer.numNeurons);
    CUDA_CHECK(cudaMemcpy(predictions.data(), outputLayer.d_activations, outputBytes, cudaMemcpyDeviceToHost));
    
    return predictions;
}

// Request raw memory from the GPU
void GpuNetwork::AllocateLayerMemory(GpuLayer& layer, int batchSize)
{
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
    CUDA_CHECK(cudaMemcpy(inputLayer.d_activations, d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(inputLayer.d_preActFreq, d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(inputLayer.d_gradX, d_batchInputGradX, batchNeuronBytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(inputLayer.d_gradY, d_batchInputGradY, batchNeuronBytes, cudaMemcpyDeviceToDevice));
    
    if (inputLayer.hasDualWeights) {
        CUDA_CHECK(cudaMemcpy(inputLayer.d_preActScale, d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice));
    }
    
    // Forward propagation loop
    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& curr = m_layers[i];
        GpuLayer& prev = m_layers[i - 1];

        // Launch the Forward Kernel for this specific layer
        RunForwardLayerGPU(
            m_batchSize, 
            curr.numNeurons, 
            prev.numNeurons,
            curr.d_weights, 
            curr.d_weightsScale, 
            curr.d_biases, 
            curr.d_biasesScale,
            prev.d_activations, 
            prev.d_gradX, 
            prev.d_gradY,
            curr.d_preActFreq, 
            curr.d_preActScale,
            curr.d_activations, 
            curr.d_gradX, 
            curr.d_gradY,
            curr.actType, 
            curr.hasDualWeights
        );
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
        m_batchSize, 
        outputLayer.numNeurons,
        outputLayer.d_activations, outputLayer.d_gradX, outputLayer.d_gradY,
        d_batchTargetAct, d_batchTargetGradX, d_batchTargetGradY,
        outputLayer.d_colorError, outputLayer.d_errorGradX, outputLayer.d_errorGradY,
        spatialLossWeight, 
        m_costType
    );
    
    // Update All Layers (Looping from Output down to Layer 0)
    for (int i = (int)m_layers.size() - 1; i > 0; --i)
    {
        GpuLayer& curr = m_layers[i];
        GpuLayer& prev = m_layers[i - 1];

        // We must wipe the previous layer's error buffers to zero before cuBLAS accumulates into them
        size_t batchNeuronBytes = m_batchSize * prev.numNeurons * sizeof(engineFloat);
        CUDA_CHECK(cudaMemset(prev.d_colorError, 0, batchNeuronBytes));
        CUDA_CHECK(cudaMemset(prev.d_errorGradX, 0, batchNeuronBytes));
        CUDA_CHECK(cudaMemset(prev.d_errorGradY, 0, batchNeuronBytes));

        RunBackwardLayerGPU(
            m_cublasHandle,
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

        // Update Weights
        RunAdamOptimizerGPU(numWeights, layer.d_weights, layer.d_deltaWeights, 
                            layer.d_m_weights, layer.d_v_weights, baseLearningRate, layer.adam_t);

        // Update Biases
        RunAdamOptimizerGPU(numBiases, layer.d_biases, layer.d_deltaBiases, 
                            layer.d_m_biases, layer.d_v_biases, baseLearningRate, layer.adam_t);

        // Update Dual Weights (if applicable)
        if (layer.hasDualWeights) {
            RunAdamOptimizerGPU(numWeights, layer.d_weightsScale, layer.d_deltaWeightsScale, 
                                layer.d_m_weightsScale, layer.d_v_weightsScale, baseLearningRate, layer.adam_t);

            RunAdamOptimizerGPU(numBiases, layer.d_biasesScale, layer.d_deltaBiasesScale, 
                                layer.d_m_biasesScale, layer.d_v_biasesScale, baseLearningRate, layer.adam_t);
        }
    }
    
    // Ensure all weight updates finish before the next forward pass starts
    CUDA_CHECK(cudaDeviceSynchronize());
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
