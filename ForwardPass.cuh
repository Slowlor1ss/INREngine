#pragma once
#include "Types.h"
#include <cublas_v2.h>

#include "GpuActivations.cuh"

// The main entry point for a single layer's forward pass
void RunLayerForwardPassGPU(
    cublasHandle_t handle,
    int batchSize,
    int numNeurons,
    int prevNeurons,
    const engineFloat* d_prevAct,
    const engineFloat* d_weights,
    const engineFloat* d_weights_scale, // Can be nullptr
    const engineFloat* d_biases,
    const engineFloat* d_biases_scale,  // Can be nullptr
    engineFloat* d_preActFreq,          // Output Z Freq
    engineFloat* d_preActScale,         // Output Z Scale
    engineFloat* d_activations,         // Output A
    GpuActType actType,
    bool hasDualWeights
);