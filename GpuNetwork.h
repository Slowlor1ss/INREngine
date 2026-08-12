#pragma once
#include "Types.h"
#include "Network.h"
#include "Layer.h"
#include "GpuActivations.cuh"
#include "GpuCostFunctions.cuh"
#include <cublas_v2.h>
#include <vector>

// Vram container for single layer
struct GpuLayer 
{
    int numNeurons = 0;
    int prevNeurons = 0;
    GpuActType actType = GpuActType::None;
    bool hasDualWeights = false;
    engineFloat learningRateMultiplier = 1.0f;

    // Persistent parameters (Size: numNeurons * prevNeurons)
    engineFloat* d_weights = nullptr;
    engineFloat* d_weightsScale = nullptr;
    engineFloat* d_biases = nullptr;
    engineFloat* d_biasesScale = nullptr;

    // Accumilated gradients (Size: numNeurons * prevNeurons)
    engineFloat* d_deltaWeights = nullptr;
    engineFloat* d_deltaWeightsScale = nullptr;
    engineFloat* d_deltaBiases = nullptr;
    engineFloat* d_deltaBiasesScale = nullptr;
    
    // Adam optimizer states
    int adam_t = 0; // The time step counter for bias correction
    
    engineFloat* d_m_weights = nullptr;
    engineFloat* d_v_weights = nullptr;
    engineFloat* d_m_biases = nullptr;
    engineFloat* d_v_biases = nullptr;

    engineFloat* d_m_weightsScale = nullptr;
    engineFloat* d_v_weightsScale = nullptr;
    engineFloat* d_m_biasesScale = nullptr;
    engineFloat* d_v_biasesScale = nullptr;

    // Forward pass buffers (Size: batchSize * numNeurons)
	engineFloat* d_preActFreq = nullptr;
	engineFloat* d_preActScale = nullptr;
	engineFloat* d_activations = nullptr;
	engineFloat* d_gradX = nullptr; // Also used as GEMM scratch for the raw Freq X-slope
	engineFloat* d_gradY = nullptr; // Also used as GEMM scratch for the raw Freq Y-slope

	// Forward pass GEMM scratch, dual-weight layers only (Size: batchSize * numNeurons)
	engineFloat* d_rawSlopeXScale = nullptr;
	engineFloat* d_rawSlopeYScale = nullptr;

    // Backward pass buffers (Size: batchSize * numNeurons)
    engineFloat* d_deltaAFreq = nullptr;
    engineFloat* d_deltaXFreq = nullptr;
    engineFloat* d_deltaYFreq = nullptr;
    engineFloat* d_deltaAScale = nullptr;
    engineFloat* d_deltaXScale = nullptr;
    engineFloat* d_deltaYScale = nullptr;
    
    // Error buffers from next layer (Size: batchSize * numNeurons)
    engineFloat* d_colorError = nullptr;
    engineFloat* d_errorGradX = nullptr;
    engineFloat* d_errorGradY = nullptr;
};

// GPU network manager
class GpuNetwork 
{
public:
    // Main Training wrapper
    // Takes Device pointers for the inputs and targets
    void TrainBatchGPU(
        const engineFloat* d_batchInputAct,
        const engineFloat* d_batchInputGradX,
        const engineFloat* d_batchInputGradY,
        const engineFloat* d_batchTargetAct,
        const engineFloat* d_batchTargetGradX,
        const engineFloat* d_batchTargetGradY,
        engineFloat spatialLossWeight,
        engineFloat learningRate
    );
    
    void DownloadParametersToCPU(Network& cpuNetwork);
    std::vector<engineFloat> PredictGPU(const engineFloat* d_inputAct, const engineFloat* d_inputGradX,
                                        const engineFloat* d_inputGradY);
    
public:
    // Pre-allocate required VRAM
    GpuNetwork(const Network& cpuNetwork, int batchSize);
    ~GpuNetwork();

    // Prevent double-free GPU memory
    GpuNetwork(const GpuNetwork&) = delete;
    GpuNetwork& operator=(const GpuNetwork&) = delete;

private:
    // Core
    void ForwardPass(const engineFloat* d_batchInputAct, const engineFloat* d_batchInputGradX, const engineFloat* d_batchInputGradY);
    void BackwardPass(const engineFloat* d_batchTargetAct, const engineFloat* d_batchTargetGradX, const engineFloat* d_batchTargetGradY, engineFloat spatialLossWeight);
    // Kept for debugging purposes
    //void TransferDeltasToCPU(Network& cpuNetwork);
    void ApplyGradientsGPU(engineFloat baseLearningRate);

private:
    void AllocateLayerMemory(GpuLayer& gpuLayer, int batchSize);
    void FreeLayerMemory(GpuLayer& gpuLayer);

    int m_batchSize;
    cublasHandle_t m_cublasHandle;
    GpuCostType m_costType;

    std::vector<GpuLayer> m_layers;
};