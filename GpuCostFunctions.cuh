#pragma once
#include "Types.h"
#include <cuda_runtime.h>
#include <cmath>

#ifdef __CUDACC__
#define __MATH_FUNC__ __host__ __device__ __forceinline__
#else
#define __MATH_FUNC__ inline
#endif

enum class GpuCostType {
    MSE = 0,
    L1 = 1,
    Charbonnier = 2,
};

namespace SharedCost 
{
    // Hyperparameters
    namespace Config
    {
        // Small epsilon in normalized [0,1] pixel space -- big enough to keep the
        // gradient well-behaved near zero error, small enough to stay close to L1.
        constexpr engineFloat Charbonnier_eps = 1e-3f;
        constexpr engineFloat Charbonnier_eps_sqrd = Charbonnier_eps * Charbonnier_eps ;
    }
    
    // MSE
    __MATH_FUNC__ engineFloat MSE(engineFloat activation, engineFloat target) {
        engineFloat diff = activation - target;
        return diff * diff;
    }
    
    __MATH_FUNC__ engineFloat MSEDeriv(engineFloat activation, engineFloat target) {
        return 2.0f * (activation - target);
    }
    
    // L1
    __MATH_FUNC__ engineFloat L1(engineFloat activation, engineFloat target) {
        return std::abs(activation - target);
    }
    
    __MATH_FUNC__ engineFloat L1Deriv(engineFloat activation, engineFloat target) {
        return (target > activation) ? -1.0f : 1.0f;
    }

    // Charbonnier
    __MATH_FUNC__ engineFloat Charbonnier(engineFloat activation, engineFloat target) {
        engineFloat diff = activation - target;
        return std::sqrt(diff * diff + Config::Charbonnier_eps_sqrd);
    }
    
    __MATH_FUNC__ engineFloat CharbonnierDeriv(engineFloat activation, engineFloat target) {
        engineFloat diff = activation - target;
        // d/d(activation) sqrt(diff^2 + eps^2) = diff / sqrt(diff^2 + eps^2)
        return diff / std::sqrt(diff * diff + Config::Charbonnier_eps_sqrd); 
    }

    
    // GPU KERNEL ROUTER
    __MATH_FUNC__ engineFloat Execute(GpuCostType costType, engineFloat activation, engineFloat target) {
        switch (costType) {
            case GpuCostType::MSE:         return MSE(activation, target);
            case GpuCostType::L1:          return L1(activation, target);
            case GpuCostType::Charbonnier: return Charbonnier(activation, target);
            default:                       return MSE(activation, target);
        }
    }
    
    __MATH_FUNC__ engineFloat ExecuteDerivative(GpuCostType costType, engineFloat activation, engineFloat target) {
        switch (costType) {
            case GpuCostType::MSE:         return MSEDeriv(activation, target);
            case GpuCostType::L1:          return L1Deriv(activation, target);
            case GpuCostType::Charbonnier: return CharbonnierDeriv(activation, target);
            default:                       return MSEDeriv(activation, target);
        }
    }
}