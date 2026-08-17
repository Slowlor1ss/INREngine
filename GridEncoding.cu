#include "GridEncoding.cuh"
#include "cublas_utils.h"

// One thread per (pixel, level): interpolate that level's F features at this
// pixel's (x,y) and write them into that level's channel slice of the flat
// encoder output buffers
// Output layout = [pixelIdx * totalChannels + channelIdx], totalChannels = numLevels*F.
__global__ void GridEncodeForwardKernel(
    int batchSize, int numLevels, int featuresPerLevel,
    const int* d_levelResolutions,   // [numLevels], precomputed on host once
    const int* d_levelParamOffsets,  // [numLevels], flat offset into d_gridParams
    const engineFloat* d_gridParams, // all levels concatenated
    const engineFloat* d_pixelX, const engineFloat* d_pixelY, // [batchSize], normalized [0,1]
    engineFloat* d_encodedAct, engineFloat* d_encodedGradX, engineFloat* d_encodedGradY)
{
    int totalWork = batchSize * numLevels;
    int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalWork; idx += stride)
    {
        int pixelIdx = idx / numLevels;
        int level = idx % numLevels;

        int resolution = d_levelResolutions[level];
        const engineFloat* levelParams = d_gridParams + d_levelParamOffsets[level];

        engineFloat x = d_pixelX[pixelIdx];
        engineFloat y = d_pixelY[pixelIdx];

        int outBase = pixelIdx * (numLevels * featuresPerLevel) + level * featuresPerLevel;

        GridEncoding::InterpolateLevel(
            levelParams, resolution, featuresPerLevel, x, y,
            &d_encodedAct[outBase], &d_encodedGradX[outBase], &d_encodedGradY[outBase]);
    }
}

// One thread per (pixel, level): scatter this level's share of the error 
// RunBackwardLayerGPU already computed for "layer 0" (d_nextColorError
// etc, sized batchSize * totalChannels) into that level's grad buffer
__global__ void GridEncodeBackwardKernel(
    int batchSize, int numLevels, int featuresPerLevel,
    const int* d_levelResolutions,
    const int* d_levelParamOffsets,
    engineFloat* d_gridGrad, // same layout as d_gridParams
    const engineFloat* d_pixelX, const engineFloat* d_pixelY,
    const engineFloat* d_nextColorError, const engineFloat* d_nextErrorGradX, const engineFloat* d_nextErrorGradY)
{
    int totalWork = batchSize * numLevels;
    int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalWork; idx += stride)
    {
        int pixelIdx = idx / numLevels;
        int level = idx % numLevels;

        int resolution = d_levelResolutions[level];
        engineFloat* levelGrad = d_gridGrad + d_levelParamOffsets[level];

        engineFloat x = d_pixelX[pixelIdx];
        engineFloat y = d_pixelY[pixelIdx];

        int errBase = pixelIdx * (numLevels * featuresPerLevel) + level * featuresPerLevel;

        GridEncoding::ScatterLevelGradient(
            levelGrad, resolution, featuresPerLevel, x, y,
            &d_nextColorError[errBase], &d_nextErrorGradX[errBase], &d_nextErrorGradY[errBase]);
    }
}

// HOST LAUNCHERS: same shape as RunForwardLayerGPU / RunBackwardLayerGPU so
// they drop into our per-batch call sequence
void RunGridEncodeForwardGPU(
    cudaStream_t stream, int batchSize, int numLevels, int featuresPerLevel,
    const int* d_levelResolutions, const int* d_levelParamOffsets,
    const engineFloat* d_gridParams,
    const engineFloat* d_pixelX, const engineFloat* d_pixelY,
    engineFloat* d_encodedAct, engineFloat* d_encodedGradX, engineFloat* d_encodedGradY)
{
    static int numSMs = 0;
    if (numSMs == 0) cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int threadsPerBlock = 256;
    int blocksPerGrid = numSMs * 4;

    GridEncodeForwardKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        batchSize, numLevels, featuresPerLevel,
        d_levelResolutions, d_levelParamOffsets, d_gridParams,
        d_pixelX, d_pixelY, d_encodedAct, d_encodedGradX, d_encodedGradY);
}

void RunGridEncodeBackwardGPU(
    cudaStream_t stream, int batchSize, int numLevels, int featuresPerLevel,
    const int* d_levelResolutions, const int* d_levelParamOffsets,
    engineFloat* d_gridGrad,
    const engineFloat* d_pixelX, const engineFloat* d_pixelY,
    const engineFloat* d_nextColorError, const engineFloat* d_nextErrorGradX, const engineFloat* d_nextErrorGradY)
{
    static int numSMs = 0;
    if (numSMs == 0) cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int threadsPerBlock = 256;
    int blocksPerGrid = numSMs * 4;

    GridEncodeBackwardKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        batchSize, numLevels, featuresPerLevel,
        d_levelResolutions, d_levelParamOffsets, d_gridGrad,
        d_pixelX, d_pixelY, d_nextColorError, d_nextErrorGradX, d_nextErrorGradY);
}
