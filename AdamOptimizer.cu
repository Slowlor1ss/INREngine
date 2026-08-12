#include "AdamOptimizer.cuh"
#include <math.h>

__global__ void AdamKernel(
    const int numElements,
    engineFloat* d_params,
    engineFloat* d_gradients,
    engineFloat* d_m,
    engineFloat* d_v,
    const engineFloat learningRate,
    const engineFloat biasCorr1, // 1 - beta1^t, computed once on host
    const engineFloat biasCorr2, // 1 - beta2^t, computed once on host
    int batchSize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    // Standard Adam Hyperparameters
    constexpr engineFloat beta1 = 0.9f;
    constexpr engineFloat beta2 = 0.999f;
    constexpr engineFloat epsilon = 1e-8f;
    
    // Average gradient across batch
    engineFloat g = d_gradients[idx] / (engineFloat)batchSize;

    engineFloat m = beta1 * d_m[idx] + (1.0f - beta1) * g;
    d_m[idx] = m;

    engineFloat v = beta2 * d_v[idx] + (1.0f - beta2) * (g * g);
    d_v[idx] = v;

    //// Compute bias-corrected moments
    //engineFloat m_hat = m / (1.0f - powf(beta1, static_cast<engineFloat>(t)));
    //engineFloat v_hat = v / (1.0f - powf(beta2, static_cast<engineFloat>(t)));

	// Apply the update to the actual parameter (Subtract to move against the error gradient)
    engineFloat m_hat = m / biasCorr1;
    engineFloat v_hat = v / biasCorr2;

    d_params[idx] -= learningRate * m_hat / (sqrtf(v_hat) + epsilon);

	// Auto-clear the gradient so it is zeroed for the next batch
    d_gradients[idx] = 0.0f;
}

void RunAdamOptimizerGPU(
    const int numElements, engineFloat* d_params, engineFloat* d_gradients,
    engineFloat* d_m, engineFloat* d_v, const engineFloat learningRate, const int t, int batchSize)
{
    if (numElements <= 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Computed once per call on the CPU instead of once per GPU thread
    const engineFloat biasCorr1 = 1.0f - powf(0.9f, static_cast<engineFloat>(t));
    const engineFloat biasCorr2 = 1.0f - powf(0.999f, static_cast<engineFloat>(t));

    AdamKernel<<<blocksPerGrid, threadsPerBlock>>>(
        numElements, d_params, d_gradients, d_m, d_v, learningRate, biasCorr1, biasCorr2, batchSize
    );
}
