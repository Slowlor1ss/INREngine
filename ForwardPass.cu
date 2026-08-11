#include "ForwardPass.cuh"
#include "cublas_utils.h"
#include <cmath>

#include "GpuActivations.cuh"

// Add biases and apply activations
__global__ void ForwardLayerKernel(
    int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_biases, const engineFloat* d_biasesScale,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    engineFloat* d_preActFreq, engineFloat* d_preActScale,
    engineFloat* d_activations, engineFloat* d_gradX, engineFloat* d_gradY,
    GpuActType actType, bool hasDualWeights)
{
    // 1 Thread = 1 Pixel in the batch
    const int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize) return;

    for (int i = 0; i < numNeurons; ++i)
    {
        // Start sums with the layer biases
        engineFloat zFreq = d_biases[i];
        engineFloat zScale = hasDualWeights ? d_biasesScale[i] : 0.0f;
        
        engineFloat gradXZFreq = 0.0f, gradXZScale = 0.0f;
        engineFloat gradYZFreq = 0.0f, gradYZScale = 0.0f;

        const int weightStart = i * prevNeurons;

        // Multiply weights by previous layer's activations and slopes
        for (int j = 0; j < prevNeurons; ++j)
        {
            const int prevIdx = batchIdx * prevNeurons + j;
            const engineFloat pA  = d_prevAct[prevIdx];
            const engineFloat pGX = d_prevGradX[prevIdx];
            const engineFloat pGY = d_prevGradY[prevIdx];

            // Freq updates
            const engineFloat wF = d_weights[weightStart + j];
            zFreq      += wF * pA;
            gradXZFreq += wF * pGX;
            gradYZFreq += wF * pGY;

            // Scale updates
            if (hasDualWeights) {
                const engineFloat wS = d_weightsScale[weightStart + j];
                zScale      += wS * pA;
                gradXZScale += wS * pGX;
                gradYZScale += wS * pGY;
            }
        }

        // Save pre-activations (required for Backward Pass)
        const int currIdx = batchIdx * numNeurons + i;
        d_preActFreq[currIdx] = zFreq;
        if (hasDualWeights) {
            d_preActScale[currIdx] = zScale;
        }

        // Execute Forward Activation & Derivatives
        d_activations[currIdx] = SharedAct::Execute(actType, zFreq, zScale);

        engineFloat deriv1Freq, deriv1Scale, deriv2Freq, deriv2Scale, deriv2Mixed;
        SharedAct::ExecuteDerivatives(actType, zFreq, zScale, deriv1Freq, deriv1Scale, deriv2Freq, deriv2Scale, deriv2Mixed);

        // Chain Rule: Multiply spatial sums by the activation's first derivative
        d_gradX[currIdx] = (gradXZFreq * deriv1Freq) + (gradXZScale * deriv1Scale);
        d_gradY[currIdx] = (gradYZFreq * deriv1Freq) + (gradYZScale * deriv1Scale);
    }
}

void RunForwardLayerGPU(
    int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_biases, const engineFloat* d_biasesScale,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    engineFloat* d_preActFreq, engineFloat* d_preActScale,
    engineFloat* d_activations, engineFloat* d_gradX, engineFloat* d_gradY,
    GpuActType actType, bool hasDualWeights)
{
    // Launch exactly one thread per pixel in the batch
    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    ForwardLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(
        batchSize, numNeurons, prevNeurons,
        d_weights, d_weightsScale, d_biases, d_biasesScale,
        d_prevAct, d_prevGradX, d_prevGradY,
        d_preActFreq, d_preActScale,
        d_activations, d_gradX, d_gradY,
        actType, hasDualWeights
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}