#pragma once
#include "../C++/Types.h"
#include <cublas_v2.h>

#include "GpuActivations.cuh"

// The main entry point for a single layer's forward pass
void RunForwardLayerGPU(
    cublasHandle_t handle, cudaStream_t stream,
    int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_biases, const engineFloat* d_biasesScale,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    engineFloat* d_preActFreq,  // Output Z Freq (also GEMM output buffer)
    engineFloat* d_preActScale, // Output Z Scale (also GEMM output buffer)
    engineFloat* d_activations, // Output A
    engineFloat* d_gradX, engineFloat* d_gradY, // Output, also doubles as raw-slope GEMM scratch
    engineFloat* d_rawSlopeXScale, engineFloat* d_rawSlopeYScale, // Scratch, dual-weight only
    GpuActType actType, bool hasDualWeights
);
