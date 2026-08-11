#pragma once
#include "Types.h"
#include <cublas_v2.h>

#include "GpuActivations.cuh"

// The main entry point for a single layer's forward pass
void RunForwardLayerGPU(
    int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_biases, const engineFloat* d_biasesScale,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    engineFloat* d_preActFreq,  // Output Z Freq
    engineFloat* d_preActScale, // Output Z Scale
    engineFloat* d_activations, // Output A
    engineFloat* d_gradX, engineFloat* d_gradY,
    GpuActType actType, bool hasDualWeights
);