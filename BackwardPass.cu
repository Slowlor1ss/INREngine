#include "BackwardPass.cuh"
#include "cublas_utils.h"
#include "GpuCostFunctions.cuh"

// OUTPUT ERROR KERNEL
__global__ void OutputErrorKernel(
    int batchSize, int numNeurons,
    const engineFloat* d_outputAct, const engineFloat* d_outputGradX, const engineFloat* d_outputGradY,
    const engineFloat* d_targetAct, const engineFloat* d_targetGradX, const engineFloat* d_targetGradY,
    engineFloat* d_colorError, engineFloat* d_errorGradX, engineFloat* d_errorGradY,
    engineFloat spatialLossWeight,
    GpuCostType costType)
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



void RunBackwardLayerGPU(
    int batchSize, int numNeurons, int prevNeurons,
    const engineFloat* d_colorErrorIn, const engineFloat* d_errorGradXIn, const engineFloat* d_errorGradYIn,
    const engineFloat* d_prevAct, const engineFloat* d_prevGradX, const engineFloat* d_prevGradY,
    const engineFloat* d_weights, const engineFloat* d_weightsScale,
    const engineFloat* d_preActFreq, const engineFloat* d_preActScale,
    engineFloat* d_nextColorError, engineFloat* d_nextErrorGradX, engineFloat* d_nextErrorGradY,
    engineFloat* d_deltaWeights, engineFloat* d_deltaWeightsScale,
    engineFloat* d_deltaBiases, engineFloat* d_deltaBiasesScale,
    GpuActType actType, bool hasDualWeights)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    BackwardLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(
        batchSize, numNeurons, prevNeurons,
        d_colorErrorIn, d_errorGradXIn, d_errorGradYIn,
        d_prevAct, d_prevGradX, d_prevGradY,
        d_weights, d_weightsScale,
        d_preActFreq, d_preActScale,
        d_nextColorError, d_nextErrorGradX, d_nextErrorGradY,
        d_deltaWeights, d_deltaWeightsScale,
        d_deltaBiases, d_deltaBiasesScale,
        actType, hasDualWeights
    );

    CUDA_CHECK(cudaDeviceSynchronize());
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
