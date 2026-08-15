#pragma once
#include "Types.h"
#include <cuda_runtime.h>
#include <cmath>

// Host launchers, implemented in GridEncoding.cu -- declared here so GpuNetwork.cpp
// can call them the same way it calls RunForwardLayerGPU/RunBackwardLayerGPU.
void RunGridEncodeForwardGPU(
    cudaStream_t stream, int batchSize, int numLevels, int featuresPerLevel,
    const int* d_levelResolutions, const int* d_levelParamOffsets,
    const engineFloat* d_gridParams,
    const engineFloat* d_pixelX, const engineFloat* d_pixelY,
    engineFloat* d_encodedAct, engineFloat* d_encodedGradX, engineFloat* d_encodedGradY);

void RunGridEncodeBackwardGPU(
    cudaStream_t stream, int batchSize, int numLevels, int featuresPerLevel,
    const int* d_levelResolutions, const int* d_levelParamOffsets,
    engineFloat* d_gridGrad,
    const engineFloat* d_pixelX, const engineFloat* d_pixelY,
    const engineFloat* d_nextColorError, const engineFloat* d_nextErrorGradX, const engineFloat* d_nextErrorGradY);

#ifdef __CUDACC__
    #define __GRID_FUNC__ __host__ __device__ __forceinline__
#else
    #define __GRID_FUNC__ inline
#endif

// Dense multi-resolution 2D feature grid encoder (Instant-NGP-style, minus the
// hash table -- our domain is one bounded image, so a small dense grid per level
// is cheap enough that we don't need to accept hash collisions to save memory).
//
// Each level owns its own R_l x R_l x F block of trainable features. A query
// coordinate (x,y) in [0,1]x[0,1] is bilinearly interpolated per level, and the
// per-level F-dim results get concatenated into the encoder's output vector --
// which is then just "d_prevAct" for your existing Layer 0, unchanged.
//
// NOTE: this file is the self-contained math core only (forward interpolation +
// exact analytic dValue/dx,dy, and the backward scatter coefficients). The
// surrounding __global__ kernels + host launchers (RunGridEncodeForwardGPU /
// RunGridEncodeBackwardGPU, mirroring ForwardPass.cu / BackwardPass.cu) and the
// GpuNetwork-side buffer management still need to be wired up against
// GpuNetwork.h/.cu, GpuDataset.h, and ImageUtils.h, which I don't have yet.
namespace GridEncoding
{
    namespace Config
    {
        // Tune to taste. Keep FinestResolution at or slightly BELOW your training
        // image's resolution -- if the grid can resolve every training pixel
        // exactly, it will happily memorize per-pixel noise, and the MLP that
        // follows loses its reason to smooth/generalize between grid cells.
        constexpr int NumLevels = 8;
        constexpr int BaseResolution = 16;
        constexpr int FeaturesPerLevel = 2;
        constexpr int FinestResolution = 256; // set to just under your training res
    }

    // Level resolutions form a geometric progression from BaseResolution to
    // FinestResolution across NumLevels steps -- coarse levels give the MLP cheap
    // global structure, fine levels give it cheap local detail.
    __GRID_FUNC__ engineFloat LevelGrowthFactor()
    {
        return std::exp(std::log(static_cast<engineFloat>(Config::FinestResolution) /
                                  static_cast<engineFloat>(Config::BaseResolution)) /
                         static_cast<engineFloat>(Config::NumLevels - 1));
    }

    __GRID_FUNC__ int LevelResolution(int level)
    {
        engineFloat r = static_cast<engineFloat>(Config::BaseResolution) *
                         std::pow(LevelGrowthFactor(), static_cast<engineFloat>(level));
        int ri = static_cast<int>(std::round(r));
        return ri < 2 ? 2 : ri;
    }

    __GRID_FUNC__ int LevelNumParams(int level)
    {
        int r = LevelResolution(level);
        return r * r * Config::FeaturesPerLevel;
    }

    // Bilinear cell lookup: given normalized x,y in [0,1] and this level's
    // resolution, find the 4 surrounding grid corners and fractional weights.
    // (Pass resolution in rather than recomputing LevelResolution() per-thread --
    // precompute it once on the host per level and pass it into the kernel.)
    __GRID_FUNC__ void FindCell(
        engineFloat x, engineFloat y, int resolution,
        int& x0, int& y0, int& x1, int& y1, engineFloat& tx, engineFloat& ty)
    {
        // Map from [-1.0, 1.0] to [0.0, 1.0]
        engineFloat nx = (x + 1.0f) * 0.5f;
        engineFloat ny = (y + 1.0f) * 0.5f;

        engineFloat cx = nx < 0.0f ? 0.0f : (nx > 1.0f ? 1.0f : nx);
        engineFloat cy = ny < 0.0f ? 0.0f : (ny > 1.0f ? 1.0f : ny);
        
        engineFloat gx = cx * static_cast<engineFloat>(resolution - 1);
        engineFloat gy = cy * static_cast<engineFloat>(resolution - 1);

        x0 = static_cast<int>(gx);
        y0 = static_cast<int>(gy);
        x1 = (x0 + 1 < resolution - 1) ? x0 + 1 : resolution - 1;
        y1 = (y0 + 1 < resolution - 1) ? y0 + 1 : resolution - 1;

        tx = gx - static_cast<engineFloat>(x0);
        ty = gy - static_cast<engineFloat>(y0);
    }

    // FORWARD: interpolate one level's F features at (x,y), plus the analytic
    // dValue/dx and dValue/dy per feature -- needed for your gradX/gradY spatial
    // loss machinery. Bilinear interpolation is piecewise-linear, so these are
    // exact closed forms; no second derivatives needed anywhere in this encoder,
    // unlike the activation functions.
    __GRID_FUNC__ void InterpolateLevel(
        const engineFloat* levelParams, int resolution, int numFeatures,
        engineFloat x, engineFloat y,
        engineFloat* outValue, engineFloat* outDx, engineFloat* outDy) // each numFeatures long
    {
        int x0, y0, x1, y1;
        engineFloat tx, ty;
        FindCell(x, y, resolution, x0, y0, x1, y1, tx, ty);

        const int i00 = (y0 * resolution + x0) * numFeatures;
        const int i10 = (y0 * resolution + x1) * numFeatures;
        const int i01 = (y1 * resolution + x0) * numFeatures;
        const int i11 = (y1 * resolution + x1) * numFeatures;

        const engineFloat w00 = (1.0f - tx) * (1.0f - ty);
        const engineFloat w10 = tx * (1.0f - ty);
        const engineFloat w01 = (1.0f - tx) * ty;
        const engineFloat w11 = tx * ty;

        // Chain rule through gx = x*(resolution-1) folds into this scale factor.
        // Because we mapped [-1, 1] to [0, 1], du/dx = 0.5.
        const engineFloat scale = static_cast<engineFloat>(resolution - 1) * 0.5f;

        for (int f = 0; f < numFeatures; ++f)
        {
            engineFloat f00 = levelParams[i00 + f];
            engineFloat f10 = levelParams[i10 + f];
            engineFloat f01 = levelParams[i01 + f];
            engineFloat f11 = levelParams[i11 + f];

            outValue[f] = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11;
            outDx[f] = scale * ((1.0f - ty) * (f10 - f00) + ty * (f11 - f01));
            outDy[f] = scale * ((1.0f - tx) * (f01 - f00) + tx * (f11 - f10));
        }
    }

    // BACKWARD: scatter this level's share of the upstream error into the 4
    // corner feature vectors read on the forward pass. Interpolation is linear in
    // the stored features, so each of the three upstream errors (color, dx, dy)
    // just distributes by its own coefficient and sums -- no second-derivative
    // term needed. Multiple query points will land in the same cell across a
    // batch, hence atomicAdd on device.
    __GRID_FUNC__ void ScatterLevelGradient(
        engineFloat* levelGrad, int resolution, int numFeatures,
        engineFloat x, engineFloat y,
        const engineFloat* errColor, const engineFloat* errGradX, const engineFloat* errGradY) // numFeatures long each
    {
        int x0, y0, x1, y1;
        engineFloat tx, ty;
        FindCell(x, y, resolution, x0, y0, x1, y1, tx, ty);

        const int i00 = (y0 * resolution + x0) * numFeatures;
        const int i10 = (y0 * resolution + x1) * numFeatures;
        const int i01 = (y1 * resolution + x0) * numFeatures;
        const int i11 = (y1 * resolution + x1) * numFeatures;

        const engineFloat w00 = (1.0f - tx) * (1.0f - ty);
        const engineFloat w10 = tx * (1.0f - ty);
        const engineFloat w01 = (1.0f - tx) * ty;
        const engineFloat w11 = tx * ty;
        // Chain rule scaling must match the forward pass so * 0.5
        const engineFloat scale = static_cast<engineFloat>(resolution - 1) * 0.5f;

        for (int f = 0; f < numFeatures; ++f)
        {
            engineFloat cErr = errColor[f];
            engineFloat xErr = errGradX[f];
            engineFloat yErr = errGradY[f];

            engineFloat g00 = cErr * w00 + scale * (xErr * -(1.0f - ty) + yErr * -(1.0f - tx));
            engineFloat g10 = cErr * w10 + scale * (xErr *  (1.0f - ty) + yErr * -tx);
            engineFloat g01 = cErr * w01 + scale * (xErr * -ty         + yErr *  (1.0f - tx));
            engineFloat g11 = cErr * w11 + scale * (xErr *  ty         + yErr *  tx);

#ifdef __CUDA_ARCH__
            atomicAdd(&levelGrad[i00 + f], g00);
            atomicAdd(&levelGrad[i10 + f], g10);
            atomicAdd(&levelGrad[i01 + f], g01);
            atomicAdd(&levelGrad[i11 + f], g11);
#else
            levelGrad[i00 + f] += g00;
            levelGrad[i10 + f] += g10;
            levelGrad[i01 + f] += g01;
            levelGrad[i11 + f] += g11;
#endif
        }
    }
}
