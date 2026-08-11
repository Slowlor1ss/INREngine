#include "AdamOptimizer.cuh"
#include <math.h>

__global__ void AdamKernel(
    const int numElements,
    engineFloat* d_params,
    engineFloat* d_gradients,
    engineFloat* d_m,
    engineFloat* d_v,
    const engineFloat learningRate,
    const int t)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    // Standard Adam Hyperparameters
    constexpr engineFloat beta1 = 0.9f;
    constexpr engineFloat beta2 = 0.999f;
    constexpr engineFloat epsilon = 1e-8f;

    engineFloat g = d_gradients[idx];

    // Update biased first moment (Momentum)
    engineFloat m = beta1 * d_m[idx] + (1.0f - beta1) * g;
    d_m[idx] = m;

    // Update biased second moment (Velocity)
    engineFloat v = beta2 * d_v[idx] + (1.0f - beta2) * (g * g);
    d_v[idx] = v;

    // Compute bias-corrected moments
    engineFloat m_hat = m / (1.0f - powf(beta1, static_cast<engineFloat>(t)));
    engineFloat v_hat = v / (1.0f - powf(beta2, static_cast<engineFloat>(t)));

    // Apply the update to the actual parameter (Subtract to move against the error gradient)
    d_params[idx] -= learningRate * m_hat / (sqrtf(v_hat) + epsilon);

    // Auto-clear the gradient so it is perfectly zeroed for the next batch!
    d_gradients[idx] = 0.0f;
}

void RunAdamOptimizerGPU(
    const int numElements, engineFloat* d_params, engineFloat* d_gradients, 
    engineFloat* d_m, engineFloat* d_v, const engineFloat learningRate, const int t)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    AdamKernel<<<blocksPerGrid, threadsPerBlock>>>(
        numElements, d_params, d_gradients, d_m, d_v, learningRate, t
    );
}