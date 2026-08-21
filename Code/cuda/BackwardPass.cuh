#pragma once
#include "Types.h"
#include "GpuActivations.cuh"
#include <cublas_v2.h>

enum class GpuCostType;

engineFloat CalculateBatchCostGPU(
    int batchSize, int numNeurons,
    const engineFloat* d_outputAct, const engineFloat* d_targetAct,
    engineFloat* d_outCost, GpuCostType costType);

// Calculate the starting errors at the output layer
void CalculateOutputErrorGPU(
    cudaStream_t stream,
    int batchSize, 
    int numNeurons,
    const engineFloat* d_outputAct,
    const engineFloat* d_outputGradX,
    const engineFloat* d_outputGradY,
    const engineFloat* d_targetAct,
    const engineFloat* d_targetGradX,
    const engineFloat* d_targetGradY,
    engineFloat* d_colorError,
    engineFloat* d_errorGradX,
    engineFloat* d_errorGradY,
    engineFloat spatialLossWeight,
    GpuCostType costType
);

// Calculate the weight/bias updates and pass the error backwards
void RunBackwardLayerGPU(
    cublasHandle_t handle, cudaStream_t stream,
    int batchSize, int numNeurons, int prevNumNeurons,
    const engineFloat* d_colorErrorIn, const engineFloat* d_errorGradXIn, const engineFloat* d_errorGradYIn,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_preActFreq, const engineFloat* d_preActScale,
    
    // Factored intermediate buffers
    engineFloat* d_deltaAFreq, engineFloat* d_deltaXFreq, engineFloat* d_deltaYFreq,
    engineFloat* d_deltaAScale, engineFloat* d_deltaXScale, engineFloat* d_deltaYScale,

    // Outputs
    engineFloat* d_nextColorError, engineFloat* d_nextErrorGradX, engineFloat* d_nextErrorGradY,
    engineFloat* d_deltaWeights, engineFloat* d_deltaWeightsScale,
    engineFloat* d_deltaBiases, engineFloat* d_deltaBiasesScale,
    
    GpuActType actType, bool hasDualWeights
);