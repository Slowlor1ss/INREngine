#include "BackwardPass.cuh"
#include "cublas_utils.h"
#include "GpuCostFunctions.cuh"

// OUTPUT ERROR KERNEL
__global__ void OutputErrorKernel(
    int batchSize, int numNeurons,
    const engineFloat* d_outputAct, const engineFloat* d_outputGradX, const engineFloat* d_outputGradY,
    const engineFloat* d_targetAct, const engineFloat* d_targetGradX, const engineFloat* d_targetGradY,
    engineFloat* d_colorError, engineFloat* d_errorGradX, engineFloat* d_errorGradY,
    engineFloat spatialLossWeight, GpuCostType costType)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * numNeurons;

    if (idx < totalElements)
    {
        // Dynamically calculate Color Error
        d_colorError[idx] = SharedCost::ExecuteDerivative(costType, d_outputAct[idx], d_targetAct[idx]);
        
        // Dynamically calculate scaled Spatial Errors
        d_errorGradX[idx] = SharedCost::ExecuteDerivative(costType, d_outputGradX[idx], d_targetGradX[idx]) * spatialLossWeight;
        d_errorGradY[idx] = SharedCost::ExecuteDerivative(costType, d_outputGradY[idx], d_targetGradY[idx]) * spatialLossWeight;
    }
}

// HOST LAUNCHER FUNCTION
void CalculateOutputErrorGPU(
    int batchSize, int numNeurons,
    const engineFloat* d_outputAct, const engineFloat* d_outputGradX, const engineFloat* d_outputGradY,
    const engineFloat* d_targetAct, const engineFloat* d_targetGradX, const engineFloat* d_targetGradY,
    engineFloat* d_colorError, engineFloat* d_errorGradX, engineFloat* d_errorGradY,
    engineFloat spatialLossWeight, GpuCostType costType)
{
    int totalElements = batchSize * numNeurons;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    OutputErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        batchSize, numNeurons,
        d_outputAct, d_outputGradX, d_outputGradY,
        d_targetAct, d_targetGradX, d_targetGradY,
        d_colorError, d_errorGradX, d_errorGradY,
        spatialLossWeight, costType
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

// FACTORED DELTA KERNEL
__global__ void ComputeDeltaTermsKernel(
    int batchSize, int numNeurons, int prevNeuronsCount,
    const engineFloat* d_colorErrorIn, const engineFloat* d_errorGradXIn, const engineFloat* d_errorGradYIn,
    const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_preActFreq, const engineFloat* d_preActScale,
    engineFloat* d_deltaAFreq, engineFloat* d_deltaXFreq, engineFloat* d_deltaYFreq,
    engineFloat* d_deltaAScale, engineFloat* d_deltaXScale, engineFloat* d_deltaYScale,
    engineFloat* d_deltaBiases, engineFloat* d_deltaBiasesScale,
    GpuActType actType, bool hasDualWeights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * numNeurons;
    
    if (idx >= totalElements) return;

    int batchIdx = idx / numNeurons;
    int neuronIdx = idx % numNeurons;

    engineFloat zFreq = d_preActFreq[idx];
    engineFloat zScale = hasDualWeights ? d_preActScale[idx] : 0.0f;
    
    // Get Derivatives
    engineFloat deriv1Freq, deriv1Scale, deriv2Freq, deriv2Scale, deriv2Mixed;
    SharedAct::ExecuteDerivatives(actType, zFreq, zScale, deriv1Freq, deriv1Scale, deriv2Freq, deriv2Scale, deriv2Mixed);

    // Calculate Raw Slopes
    engineFloat rawSlopeXFreq = 0.0f, rawSlopeXScale = 0.0f;
    engineFloat rawSlopeYFreq = 0.0f, rawSlopeYScale = 0.0f;
    int weightStart = neuronIdx * prevNeuronsCount;

    for (int j = 0; j < prevNeuronsCount; ++j) {
        int prevIdx = batchIdx * prevNeuronsCount + j;
        engineFloat wF = d_weights[weightStart + j]; 
        rawSlopeXFreq += wF * d_prevGradX[prevIdx];
        rawSlopeYFreq += wF * d_prevGradY[prevIdx];

        if (hasDualWeights) {
            engineFloat wS = d_weightsScale[weightStart + j]; 
            rawSlopeXScale += wS * d_prevGradX[prevIdx];
            rawSlopeYScale += wS * d_prevGradY[prevIdx];
        }
    }

    engineFloat cErr = d_colorErrorIn[idx];
    engineFloat xErr = d_errorGradXIn[idx];
    engineFloat yErr = d_errorGradYIn[idx];

    // Compute Factored Terms
    engineFloat deltaAF = cErr * deriv1Freq + xErr * (deriv2Freq * rawSlopeXFreq + deriv2Mixed * rawSlopeXScale) 
                                            + yErr * (deriv2Freq * rawSlopeYFreq + deriv2Mixed * rawSlopeYScale);
    engineFloat deltaXF = xErr * deriv1Freq;
    engineFloat deltaYF = yErr * deriv1Freq;

    d_deltaAFreq[idx] = deltaAF;
    d_deltaXFreq[idx] = deltaXF;
    d_deltaYFreq[idx] = deltaYF;
    
    // deltaA is mathematically identical to the bias gradient (should be at least...)
    atomicAdd(&d_deltaBiases[neuronIdx], deltaAF);

    if (hasDualWeights) {
        engineFloat deltaAS = cErr * deriv1Scale + xErr * (deriv2Scale * rawSlopeXScale + deriv2Mixed * rawSlopeXFreq) 
                                                 + yErr * (deriv2Scale * rawSlopeYScale + deriv2Mixed * rawSlopeYFreq);
        engineFloat deltaXS = xErr * deriv1Scale;
        engineFloat deltaYS = yErr * deriv1Scale;
        
        d_deltaAScale[idx] = deltaAS;
        d_deltaXScale[idx] = deltaXS;
        d_deltaYScale[idx] = deltaYS;
        
        atomicAdd(&d_deltaBiasesScale[neuronIdx], deltaAS);
    }
}

// CUBLAS matrix calcualtions
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// cuBLAS reads C++ row-major memory sideways (column-major). 
// which allows multiplication of the entire batch simultaneously without loops
//
// Example:
// We are calculating: d_deltaWeights += (PrevAct) * (DeltaFreq)^T 
// (and we have 2 layers one of 2 neurons (matrix A) and one of 3 neurons (Matrix B))
//
//       [ Matrix A: d_prevAct ]          [ Matrix B^T: d_deltaAFreq^T ]     [ Matrix C: d_deltaWeights ]
//       (OP_N: Normal)                   (OP_T: Transposed)                 
//
//          (batchSize columns)               (numNeurons columns)                  (numNeurons columns)
//          pix_0    pix_1    pix_2           n_0      n_1      n_2                 n_0      n_1      n_2 
//        +------------------------+        +------------------------+           +------------------------+
// prev_0 |  a00      a01      a02 |   px_0 |  d00      d01      d02 |    prev_0 |  w00      w01      w02 |
// prev_1 |  a10      a11      a12 | x px_1 |  d10      d11      d12 | += prev_1 |  w10      w11      w12 |
//        +------------------------+   px_2 |  d20      d21      d22 |           +------------------------+
//  ^(prevNeurons rows)                      +------------------------+      ^(prevNeurons rows)
//                                      ^(batchSize rows)
//
// HOW THE BATCH COLLAPSES:
// To calculate the gradient for weight w00, cuBLAS takes the entire p0 row of Matrix A 
// (the activation of previous neuron 0 across ALL pixels) and takes the dot product with 
// the entire n0 column of Matrix B (the delta of current neuron 0 across ALL pixels). 
// The batchSize dimension collapses, leaving a single accumulated gradient
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void RunBackwardLayerGPU(
    cublasHandle_t handle,
    int batchSize, int numNeurons, int prevNumNeurons,
    const engineFloat* d_colorErrorIn, const engineFloat* d_errorGradXIn, const engineFloat* d_errorGradYIn,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_preActFreq, const engineFloat* d_preActScale,
    engineFloat* d_deltaAFreq, engineFloat* d_deltaXFreq, engineFloat* d_deltaYFreq,
    engineFloat* d_deltaAScale, engineFloat* d_deltaXScale, engineFloat* d_deltaYScale,
    engineFloat* d_nextColorError, engineFloat* d_nextErrorGradX, engineFloat* d_nextErrorGradY,
    engineFloat* d_deltaWeights, engineFloat* d_deltaWeightsScale,
    engineFloat* d_deltaBiases, engineFloat* d_deltaBiasesScale,
    GpuActType actType, bool hasDualWeights)
{
    int totalElements = batchSize * numNeurons;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate Deltas
    ComputeDeltaTermsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        batchSize, numNeurons, prevNumNeurons,
        d_colorErrorIn, d_errorGradXIn, d_errorGradYIn,
        d_prevGradX, d_prevGradY,
        d_weights, d_weightsScale, d_preActFreq, d_preActScale,
        d_deltaAFreq, d_deltaXFreq, d_deltaYFreq,
        d_deltaAScale, d_deltaXScale, d_deltaYScale,
        d_deltaBiases, d_deltaBiasesScale,
        actType, hasDualWeights
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // cublas Matrix Multiplications (dW = Delta * Prev^T)
    const engineFloat alpha = 1.0f;
    const engineFloat betaAccumulate = 1.0f; 

    // Weight Updates (Freq) - Accumulates into d_deltaWeights
    // matrix x matrix multiplications:
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, prevNumNeurons, numNeurons, batchSize,
        &alpha, d_prevAct, prevNumNeurons, d_deltaAFreq, numNeurons, &betaAccumulate, d_deltaWeights, prevNumNeurons));
        
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, prevNumNeurons, numNeurons, batchSize,
        &alpha, d_prevGradX, prevNumNeurons, d_deltaXFreq, numNeurons, &betaAccumulate, d_deltaWeights, prevNumNeurons));
        
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, prevNumNeurons, numNeurons, batchSize,
        &alpha, d_prevGradY, prevNumNeurons, d_deltaYFreq, numNeurons, &betaAccumulate, d_deltaWeights, prevNumNeurons));

    if (hasDualWeights) {
        // Weight Updates (Scale) - Accumulates into d_deltaWeightsScale
        // more matrix x matrix multiplications:
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, prevNumNeurons, numNeurons, batchSize,
            &alpha, d_prevAct, prevNumNeurons, d_deltaAScale, numNeurons, &betaAccumulate, d_deltaWeightsScale, prevNumNeurons));
            
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, prevNumNeurons, numNeurons, batchSize,
            &alpha, d_prevGradX, prevNumNeurons, d_deltaXScale, numNeurons, &betaAccumulate, d_deltaWeightsScale, prevNumNeurons));
            
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, prevNumNeurons, numNeurons, batchSize,
            &alpha, d_prevGradY, prevNumNeurons, d_deltaYScale, numNeurons, &betaAccumulate, d_deltaWeightsScale, prevNumNeurons));
    }

    // Backpropagate Errors to Previous Layer (NextError = W^T * Delta)
    // d_nextError buffers must be zeroed before this function is called!
    const engineFloat betaOverwrite = 0.0f;
    
    // nextColorError = W^T * deltaA (We overwrite the buffer, hence beta = 0.0)
    // even more matrix x matrix multiplications:
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, prevNumNeurons, batchSize, numNeurons,
        &alpha, d_weights, prevNumNeurons, d_deltaAFreq, numNeurons, &betaOverwrite, d_nextColorError, prevNumNeurons));
        
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, prevNumNeurons, batchSize, numNeurons,
        &alpha, d_weights, prevNumNeurons, d_deltaXFreq, numNeurons, &betaOverwrite, d_nextErrorGradX, prevNumNeurons));
        
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, prevNumNeurons, batchSize, numNeurons,
        &alpha, d_weights, prevNumNeurons, d_deltaYFreq, numNeurons, &betaOverwrite, d_nextErrorGradY, prevNumNeurons));

    if (hasDualWeights) {
        // If dual weights exist, accumulate the scale side into the same nextError buffers!
        // final matrix x matrix multiplications:
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, prevNumNeurons, batchSize, numNeurons,
            &alpha, d_weightsScale, prevNumNeurons, d_deltaAScale, numNeurons, &betaAccumulate, d_nextColorError, prevNumNeurons));
            
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, prevNumNeurons, batchSize, numNeurons,
            &alpha, d_weightsScale, prevNumNeurons, d_deltaXScale, numNeurons, &betaAccumulate, d_nextErrorGradX, prevNumNeurons));
            
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, prevNumNeurons, batchSize, numNeurons,
            &alpha, d_weightsScale, prevNumNeurons, d_deltaYScale, numNeurons, &betaAccumulate, d_nextErrorGradY, prevNumNeurons));
    }
}

// vvv For reference vvv
//
// BACKWARD LAYER KERNEL
// Naive implementation
// __global__ void BackwardLayerKernel(
//     int batchSize, int numNeurons, int prevNeuronsCount,
//     const engineFloat* d_colorErrorIn, const engineFloat* d_errorGradXIn, const engineFloat* d_errorGradYIn,
//     const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
//     const engineFloat* d_weights, const engineFloat* d_weightsScale,
//     const engineFloat* d_preActFreq, const engineFloat* d_preActScale,
//     engineFloat* d_nextColorError, engineFloat* d_nextErrorGradX, engineFloat* d_nextErrorGradY,
//     engineFloat* d_deltaWeights, engineFloat* d_deltaWeightsScale,
//     engineFloat* d_deltaBiases, engineFloat* d_deltaBiasesScale,
//     GpuActType actType, bool hasDualWeights)
// {
//     int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (batchIdx >= batchSize) return;
//
//     // Loop over every neuron in the current layer
//     for (int i = 0; i < numNeurons; ++i)
//     {
//         int currIdx = batchIdx * numNeurons + i;
//         
//         engineFloat zFreq = d_preActFreq[currIdx];
//         engineFloat zScale = hasDualWeights ? d_preActScale[currIdx] : 0.0f;
//         
//         // Get Activation Derivatives
//         engineFloat deriv1Freq, deriv1Scale;
//         engineFloat deriv2Freq, deriv2Scale, deriv2Mixed;
//         SharedAct::GetDerivatives(actType, zFreq, zScale, deriv1Freq, deriv1Scale, deriv2Freq, deriv2Scale, deriv2Mixed);
//
//         // Calculate Raw Slopes (Matrix-Vector multiply per pixel)
//         engineFloat rawSlopeXFreq = 0.0f, rawSlopeXScale = 0.0f;
//         engineFloat rawSlopeYFreq = 0.0f, rawSlopeYScale = 0.0f;
//         int weightStart = i * prevNeuronsCount;
//
//         for (int j = 0; j < prevNeuronsCount; ++j) {
//             int prevIdx = batchIdx * prevNeuronsCount + j;
//             engineFloat wF = d_weights[weightStart + j]; // Weight frequency
//             
//             rawSlopeXFreq += wF * d_prevGradX[prevIdx];
//             rawSlopeYFreq += wF * d_prevGradY[prevIdx];
//
//             if (hasDualWeights) {
//                 engineFloat wS = d_weightsScale[weightStart + j]; // Weight scale
//                 rawSlopeXScale += wS * d_prevGradX[prevIdx];
//                 rawSlopeYScale += wS * d_prevGradY[prevIdx];
//             }
//         }
//
//         // Accumulate Gradients for Weights and Biases
//         engineFloat cErr = d_colorErrorIn[currIdx];
//         engineFloat xErr = d_errorGradXIn[currIdx];
//         engineFloat yErr = d_errorGradYIn[currIdx];
//
//         // Biases
//         engineFloat gradBiasFreq = cErr * deriv1Freq + xErr * (deriv2Freq * rawSlopeXFreq + deriv2Mixed * rawSlopeXScale) 
//                                     + yErr * (deriv2Freq * rawSlopeYFreq + deriv2Mixed * rawSlopeYScale);
//         atomicAdd(&d_deltaBiases[i], gradBiasFreq); // Safely add to the global pool
//
//         if (hasDualWeights) {
//             engineFloat gradBiasScale = cErr * deriv1Scale + xErr * (deriv2Scale * rawSlopeXScale + deriv2Mixed * rawSlopeXFreq) 
//                                         + yErr * (deriv2Scale * rawSlopeYScale + deriv2Mixed * rawSlopeYFreq);
//             atomicAdd(&d_deltaBiasesScale[i], gradBiasScale);
//         }
//
//         // Weights & Passing errors back to previous layer
//         for (int j = 0; j < prevNeuronsCount; ++j) {
//             int prevIdx = batchIdx * prevNeuronsCount + j;
//             engineFloat pA = d_prevAct[prevIdx];
//             engineFloat pGX = d_prevGradX[prevIdx];
//             engineFloat pGY = d_prevGradY[prevIdx];
//             
//             engineFloat gradWFreq = cErr * (deriv1Freq * pA) + 
//                                     xErr * (deriv2Freq * pA * rawSlopeXFreq + deriv2Mixed * pA * rawSlopeXScale + deriv1Freq * pGX) + 
//                                     yErr * (deriv2Freq * pA * rawSlopeYFreq + deriv2Mixed * pA * rawSlopeYScale + deriv1Freq * pGY);
//             
//             atomicAdd(&d_deltaWeights[weightStart + j], gradWFreq);
//
//             if (hasDualWeights) {
//                 engineFloat gradWScale = cErr * (deriv1Scale * pA) + 
//                                          xErr * (deriv2Mixed * pA * rawSlopeXFreq + deriv2Scale * pA * rawSlopeXScale + deriv1Scale * pGX) + 
//                                          yErr * (deriv2Mixed * pA * rawSlopeYFreq + deriv2Scale * pA * rawSlopeYScale + deriv1Scale * pGY);
//                 atomicAdd(&d_deltaWeightsScale[weightStart + j], gradWScale);
//             }
//
//             engineFloat oldWeightFreq = d_weights[weightStart + j];
//             engineFloat oldWeightScale = hasDualWeights ? d_weightsScale[weightStart + j] : 0.0f;
//             engineFloat nextErrC = cErr * (deriv1Freq * oldWeightFreq + deriv1Scale * oldWeightScale) + 
//                                    xErr * (deriv2Freq * oldWeightFreq * rawSlopeXFreq + deriv2Scale * oldWeightScale * rawSlopeXScale) + 
//                                    yErr * (deriv2Freq * oldWeightFreq * rawSlopeYFreq + deriv2Scale * oldWeightScale * rawSlopeYScale);
//                                    
//             engineFloat nextErrX = xErr * (deriv1Freq * oldWeightFreq + deriv1Scale * oldWeightScale);
//             engineFloat nextErrY = yErr * (deriv1Freq * oldWeightFreq + deriv1Scale * oldWeightScale);
//
//             // As a single GPU thread owns this entire pixel (batchIdx), it should be thread safe
//             // so we don't need atomicAdd
//             d_nextColorError[prevIdx] += nextErrC;
//             d_nextErrorGradX[prevIdx] += nextErrX;
//             d_nextErrorGradY[prevIdx] += nextErrY;
//         }
//     }
// }
