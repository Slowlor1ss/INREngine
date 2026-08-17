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

// Vram container for the multi-resolution grid encoder (all levels concatenated
// into single flat buffers, indexed via d_levelResolutions/d_levelParamOffsets).
// This plays the exact same role for the grid encoder that GpuLayer plays for a
// normal dense layer: persistent params + accumulated grad + Adam state.
struct GpuGridEncoder
{
    int numLevels = 0;
    int featuresPerLevel = 0;
    int totalParams = 0;       // sum over levels of R_l*R_l*F
    int totalOutputChannels = 0; // numLevels * featuresPerLevel -- this becomes Layer 0's numNeurons

    // One entry per level, uploaded once at construction (never changes -- fine
    // to bake into the captured graph since these are fixed device pointers to
    // fixed data, same as e.g. curr.d_weights for a normal layer).
    int* d_levelResolutions = nullptr;
    int* d_levelParamOffsets = nullptr;

    // Persistent parameters + Adam state (Size: totalParams)
    engineFloat* d_params = nullptr;
    engineFloat* d_grad = nullptr; // must start at 0, auto-cleared by Adam after each apply, same as GpuLayer's deltas
    engineFloat* d_m = nullptr;
    engineFloat* d_v = nullptr;
    int adam_t = 0;
    engineFloat learningRateMultiplier = 1.0f;

    // Fixed mailbox for this batch's pixel coordinates (Size: batchSize).
    // IMPORTANT (this is the CUDA graph subtlety): the dataset's d_pixelX/d_pixelY
    // + offset is a DIFFERENT pointer value every batch, same as d_batchInputAct
    // is today. If a captured graph node referenced that varying pointer directly,
    // every replay after the first would silently keep using whatever offset was
    // current at capture time. So exactly like the existing input-layer mailbox
    // copy in TrainBatchGPU, the varying dataset pointer gets cudaMemcpyAsync'd
    // into this FIXED buffer every call (outside/before the `if (!m_graphCaptured)`
    // block, so it re-runs on every launch), and the captured
    // RunGridEncodeForwardGPU/BackwardGPU calls always read from this fixed address.
    engineFloat* d_batchPixelX = nullptr;
    engineFloat* d_batchPixelY = nullptr;
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
        // Pass gpuData.d_pixelX + offset / gpuData.d_pixelY + offset here, same
        // convention as d_batchInputAct -- the varying pointer, not a fixed one.
        // TrainBatchGPU copies it into the fixed mailbox internally.
        const engineFloat* d_batchPixelXSrc,
        const engineFloat* d_batchPixelYSrc,
        engineFloat spatialLossWeight,
        engineFloat learningRate
    );
    
    void DownloadParametersToCPU(Network& cpuNetwork);
    // std::vector<engineFloat> PredictGPU(const engineFloat* d_inputAct, const engineFloat* d_inputGradX,
    //                                     const engineFloat* d_inputGradY);
    
    engineFloat GetLastBatchCost(const engineFloat* d_batchTargetAct);
    void PredictGPU(const engineFloat* d_predictPixelX, const engineFloat* d_predictPixelY, const engineFloat* d_standardInputs, engineFloat* d_predictOutputs, int totalPixels);
    
public:
    // Pre-allocate required VRAM
    // useGridEncoding: if true, Layer 0's numNeurons MUST already equal
    // GridEncoding::Config::NumLevels * FeaturesPerLevel on the CPU side (see the
    // NeuralImageRecreator.hpp note) -- the grid encoder produces Layer 0's
    // activations every batch instead of them being a static copy-in.
    GpuNetwork(const Network& cpuNetwork, int batchSize, bool useGridEncoding);
    ~GpuNetwork();

    // Prevent double-free GPU memory
    GpuNetwork(const GpuNetwork&) = delete;
    GpuNetwork& operator=(const GpuNetwork&) = delete;

private:
    // Core
    void ForwardPass();
    void BackwardPass(engineFloat spatialLossWeight);
    // Kept for debugging purposes
    //void TransferDeltasToCPU(Network& cpuNetwork);
    void ApplyGradientsGPU(engineFloat baseLearningRate);

private:
    void AllocateLayerMemory(GpuLayer& gpuLayer, int batchSize);
    void FreeLayerMemory(GpuLayer& gpuLayer);

    void AllocateGridEncoderMemory(int batchSize);
    void FreeGridEncoderMemory();

private:
    int m_batchSize;
    cublasHandle_t m_cublasHandle;
    GpuCostType m_costType;

    // Stream & Graph State
    cudaStream_t m_stream = nullptr;
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_graphCaptured = false;

    // Dynamic Graph Parameters
    engineFloat* d_learningRate = nullptr;
    engineFloat* h_learningRate = nullptr; // Host pointer (aka just a normal cpu pointer)
    
    engineFloat* d_fixedTargetAct = nullptr;
    engineFloat* d_fixedTargetGradX = nullptr;
    engineFloat* d_fixedTargetGradY = nullptr;

    std::vector<GpuLayer> m_layers;

    bool m_useGridEncoding = false;
    GpuGridEncoder m_gridEncoder;
    
    engineFloat* d_totalCost;
};