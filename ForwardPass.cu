#include "ForwardPass.cuh"
#include "cublas_utils.h"
#include <cmath>

#include "GpuActivations.cuh"

// RAW CUDA KERNEL: Adds biases and applies activations
__global__ void ApplyActivationKernel(
    int batchSize, int numNeurons,
    const engineFloat* d_preActFreqIn, const engineFloat* d_preActScaleIn,
    const engineFloat* d_biases, const engineFloat* d_biasesScale,
    engineFloat* d_preActFreqOut, engineFloat* d_preActScaleOut,
    engineFloat* d_activations,
    GpuActType actType, bool hasDualWeights)
{
    // Calculate which exact pixel and neuron this specific GPU thread is responsible for
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * numNeurons;

    if (idx < totalElements)
    {
        // Find which neuron this is so we can apply the correct bias
        int neuronIdx = idx % numNeurons;

        // Add the Bias
        engineFloat zFreq = d_preActFreqIn[idx] + d_biases[neuronIdx];
        engineFloat zScale = 0.0f;
        
        if (hasDualWeights && d_preActScaleIn != nullptr && d_biasesScale != nullptr) {
            zScale = d_preActScaleIn[idx] + d_biasesScale[neuronIdx];
        }

        // Save the pre-activations (Z) for Backprop
        d_preActFreqOut[idx] = zFreq;
        if (hasDualWeights) d_preActScaleOut[idx] = zScale;

        // Apply the Activation Function
        engineFloat act = zFreq; // Default to 'None'

        switch (actType)
        {
            case GpuActType::Wire:      act = SharedAct::Wire(zFreq, zScale); break;
            case GpuActType::Siren:     act = SharedAct::Siren(zFreq); break;
            case GpuActType::ReLU:      act = SharedAct::ReLU(zFreq); break;
            case GpuActType::LeakyReLU: act = SharedAct::LeakyReLU(zFreq); break;
            case GpuActType::Sigmoid:   act = SharedAct::Sigmoid(zFreq); break;
            case GpuActType::Tanh:      act = SharedAct::Tanh(zFreq); break;
            case GpuActType::None:
            default:                    act = SharedAct::None(zFreq); break;
        }
        
        d_activations[idx] = act;
    }
}


// cuBLAS MANAGER: Handles the Matrix Math and launches the Kernel above
void RunLayerForwardPassGPU(
    cublasHandle_t handle,
    int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_prevAct, const engineFloat* d_weights, const engineFloat* d_weights_scale,
    const engineFloat* d_biases, const engineFloat* d_biases_scale,
    engineFloat* d_preActFreq, engineFloat* d_preActScale, engineFloat* d_activations,
    GpuActType actType, bool hasDualWeights)
{
    const engineFloat alpha = 1.0f;
    const engineFloat beta = 0.0f; // Overwrite the output buffer completely

    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference
    // cuBLAS Matrix Multiplication (Z_freq = W * A_prev)
    // Row-Major to Col-Major translation: Z^T = W * A^T
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // C++ Stores Row-Major order, but cuBLAS reads matrices Column-Major order so we need these flags
        numNeurons, batchSize, prevNeurons, // m = Number of rows of matrix op(A) and C. // n = Number of columns of matrix op(B) and C. // k = Number of columns of op(A) and rows of op(B).
        &alpha, // <type> scalar used for multiplication.
        d_weights, prevNeurons,
        d_prevAct, prevNeurons,
        &beta,
        d_preActFreq, numNeurons
    ));

    // Dual-Weight Multiplication (Z_scale = W_scale * A_prev)
    if (hasDualWeights && d_weights_scale != nullptr)
    {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            numNeurons, batchSize, prevNeurons,
            &alpha,
            d_weights_scale, prevNeurons,
            d_prevAct, prevNeurons,
            &beta,
            d_preActScale, numNeurons
        ));
    }

    // Launch the Custom Activation Kernel
    int totalElements = batchSize * numNeurons;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    ApplyActivationKernel<<<blocksPerGrid, threadsPerBlock>>>(
        batchSize, numNeurons,
        d_preActFreq, d_preActScale,
        d_biases, d_biases_scale,
        d_preActFreq, d_preActScale,
        d_activations,
        actType, hasDualWeights
    );

    // Ensure the GPU finishes the kernel before the CPU tries to move to the next layer
    CUDA_CHECK(cudaDeviceSynchronize());
}