#pragma once
#include "CudaManager.cuh"
#include "../C++/Types.h"

void RunAdamOptimizerGPU(
    cudaStream_t stream,
    const int numElements,
    engineFloat* d_params,
    engineFloat* d_gradients,
    engineFloat* d_m,
    engineFloat* d_v,
    const engineFloat* d_learningRate,
    const engineFloat lrMultiplier,
    const int t, int batchSize
);