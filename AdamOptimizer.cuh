#pragma once
#include "Types.h"

void RunAdamOptimizerGPU(
    const int numElements,
    engineFloat* d_params,
    engineFloat* d_gradients,
    engineFloat* d_m,
    engineFloat* d_v,
    engineFloat learningRate,
    const int t, int batchSize
);