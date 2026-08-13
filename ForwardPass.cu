#include "ForwardPass.cuh"
#include "cublas_utils.h"
#include <cmath>

#include "GpuActivations.cuh"


// One thread per (batch, neuron) element
// Add biases and apply activations
__global__ void ForwardFinishKernel(
    int batchSize, int numNeurons,
    const engineFloat* d_biases, const engineFloat* d_biasesScale,
    engineFloat* d_preActFreq, engineFloat* d_preActScale,
    engineFloat* d_activations, engineFloat* d_gradX, engineFloat* d_gradY,
    const engineFloat* d_rawSlopeXScale, const engineFloat* d_rawSlopeYScale,
    GpuActType actType, bool hasDualWeights)
{
    int totalElements = batchSize * numNeurons;
    int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalElements; idx += stride)
    {
        int neuronIdx = idx % numNeurons;

        // Add bias (GEMM output was the bias-free sum)
        engineFloat zFreq = d_preActFreq[idx] + d_biases[neuronIdx];
        engineFloat zScale = 0.0f;
        if (hasDualWeights) {
            zScale = d_preActScale[idx] + d_biasesScale[neuronIdx];
            d_preActScale[idx] = zScale;
        }
        d_preActFreq[idx] = zFreq;

        // Activation + derivatives
        d_activations[idx] = SharedAct::Execute(actType, zFreq, zScale);

        engineFloat deriv1Freq, deriv1Scale, deriv2Freq, deriv2Scale, deriv2Mixed;
        SharedAct::ExecuteDerivatives(actType, zFreq, zScale, deriv1Freq, deriv1Scale, deriv2Freq, deriv2Scale, deriv2Mixed);

        // Raw slopes were parked in d_gradX/d_gradY (Freq) and the scale scratch buffers by the GEMMs
        engineFloat rawXFreq = d_gradX[idx];
        engineFloat rawYFreq = d_gradY[idx];
        engineFloat rawXScale = hasDualWeights ? d_rawSlopeXScale[idx] : 0.0f;
        engineFloat rawYScale = hasDualWeights ? d_rawSlopeYScale[idx] : 0.0f;

        // Chain rule (overwrite in place - safe, this thread owns idx exclusively)
        d_gradX[idx] = (rawXFreq * deriv1Freq) + (rawXScale * deriv1Scale);
        d_gradY[idx] = (rawYFreq * deriv1Freq) + (rawYScale * deriv1Scale);
    }
}

// Computes C(numNeurons x batchSize) = W^T_cm * B_cm, which lands in row-major
// [batchSize x numNeurons] layout - i.e. exactly d_out[batchIdx*numNeurons+i].
// Used for the Z sums (B = prevAct) and the raw slope sums (B = prevGradX/Y).
static void RunWeightGemm(
    cublasHandle_t handle, int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_weights, const engineFloat* d_prevBuffer, engineFloat* d_out)
{
    const engineFloat alpha = 1.0f;
    const engineFloat beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        numNeurons, batchSize, prevNeurons,
        &alpha,
        d_weights, prevNeurons,
        d_prevBuffer, prevNeurons,
        &beta,
        d_out, numNeurons));
}

void RunForwardLayerGPU(
    cublasHandle_t handle, cudaStream_t stream,
    int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_biases, const engineFloat* d_biasesScale,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    engineFloat* d_preActFreq, engineFloat* d_preActScale,
    engineFloat* d_activations, engineFloat* d_gradX, engineFloat* d_gradY,
    engineFloat* d_rawSlopeXScale, engineFloat* d_rawSlopeYScale,
    GpuActType actType, bool hasDualWeights)
{
    // Freq side: Z, raw X slope, raw Y slope
    RunWeightGemm(handle, batchSize, numNeurons, prevNeurons, d_weights, d_prevAct,   d_preActFreq);
    RunWeightGemm(handle, batchSize, numNeurons, prevNeurons, d_weights, d_prevGradX, d_gradX);
    RunWeightGemm(handle, batchSize, numNeurons, prevNeurons, d_weights, d_prevGradY, d_gradY);

    // Scale side
    if (hasDualWeights) {
        RunWeightGemm(handle, batchSize, numNeurons, prevNeurons, d_weightsScale, d_prevAct,   d_preActScale);
        RunWeightGemm(handle, batchSize, numNeurons, prevNeurons, d_weightsScale, d_prevGradX, d_rawSlopeXScale);
        RunWeightGemm(handle, batchSize, numNeurons, prevNeurons, d_weightsScale, d_prevGradY, d_rawSlopeYScale);
    }

    // Bias + activation + chain rule, element wise
    static int numSMs = 0;
    if (numSMs == 0) {
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    }
    int threadsPerBlock = 256;
    int blocksPerGrid = numSMs * 4;

    ForwardFinishKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        batchSize, numNeurons,
        d_biases, d_biasesScale,
        d_preActFreq, d_preActScale,
        d_activations, d_gradX, d_gradY,
        d_rawSlopeXScale, d_rawSlopeYScale,
        actType, hasDualWeights
    );
}
