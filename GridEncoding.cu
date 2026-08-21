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

// Stateless per-thread hash RNG, no nvidia cuRAND state to persist across launches
// deterministic given (idx, seed), reseeded every batch via counter
__GRID_FUNC__ void HashJitter2D(unsigned int idx, unsigned int seed, float& outA, float& outB)
{
    unsigned int h = idx * 747796405u + seed * 2891336453u + 1u;
    h = (h ^ (h >> 16)) * 2246822519u;
    unsigned int h2 = (h ^ (h >> 13)) * 3266489917u;
    h ^= h >> 16;
    h2 ^= h2 >> 16;

    outA = (static_cast<float>(h  & 0x00FFFFFFu) / static_cast<float>(0x00FFFFFFu)) * 2.0f - 1.0f;
    outB = (static_cast<float>(h2 & 0x00FFFFFFu) / static_cast<float>(0x00FFFFFFu)) * 2.0f - 1.0f;
}

// Writes jittered pixel coordinates into the grid-encoder mailbox, same
// role as the plain cudaMemcpyAsync we had, only the query location moves, so the
// network is asked to hit the same noisy value from a slightly different
// coordinate every batch, the least-loss solution becomes the local average
__global__ void JitterPixelCoordsKernel(
    const engineFloat* __restrict__ srcX, const engineFloat* __restrict__ srcY,
    engineFloat* __restrict__ dstX, engineFloat* __restrict__ dstY,
    int batchSize, engineFloat jitterAmpX, engineFloat jitterAmpY, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    float jx, jy;
    HashJitter2D(static_cast<unsigned int>(idx), seed, jx, jy);

    dstX[idx] = srcX[idx] + jx * jitterAmpX;
    dstY[idx] = srcY[idx] + jy * jitterAmpY;
}

void RunJitterPixelCoordsGPU(
    const engineFloat* d_srcX, const engineFloat* d_srcY,
    engineFloat* d_dstX, engineFloat* d_dstY,
    int batchSize, engineFloat jitterAmpX, engineFloat jitterAmpY,
    unsigned int seed, cudaStream_t stream)
{
    constexpr int threads = 256;
    int blocks = (batchSize + threads - 1) / threads;
    JitterPixelCoordsKernel<<<blocks, threads, 0, stream>>>(
        d_srcX, d_srcY, d_dstX, d_dstY, batchSize, jitterAmpX, jitterAmpY, seed);
}
