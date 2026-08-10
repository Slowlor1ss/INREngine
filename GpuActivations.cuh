#pragma once
#include <algorithm>
#include "Types.h"
#include <cuda_runtime.h>
#include <cmath>

#ifdef __CUDACC__
    #define __MATH_FUNC__ __host__ __device__ __forceinline__
#else
    #define __MATH_FUNC__ inline
#endif

// We need a simple enum because C++ virtual functions cannot run on the GPU
enum class GpuActType {
    None = 0,
    Sigmoid = 1,
    ReLU = 2,
    LeakyReLU = 3,
    Siren = 4,
    Tanh = 5,
    Wire = 6,
    Finer = 7,
};

// TODO: really we should make these functions __global__ or something and replace out cpu code to also use these for when running with --use_gpu off
// right now i did a more or less lazy solution of calling the functions in ActivationFunctions.h but this could all be a lot cleaner
namespace SharedAct 
{
    // Hyperparameters
    namespace Config 
    {
        constexpr engineFloat Siren_w0 = 30.0f;
        
        constexpr engineFloat Wire_w0 = 20.0f;
        constexpr engineFloat Wire_s = 30.0f;
        constexpr engineFloat Wire_s_squared = Wire_s * Wire_s;
        
        constexpr engineFloat LeakyReLU_slope = 0.1f;

        // FINER (variable-periodic sine): sin(w0 * (|x|+1) * x)
        constexpr engineFloat Finer_w0 = 30.0f;
        // Bias init range for FINER -- this is what actually gives it its extra
        // frequency range over SIREN, see GenerateInitialBiases below.
        constexpr engineFloat Finer_bias_k = 2.5f;
    }
    
    // Forward Pass Activation Functions

    // NONE (Linear)
    __MATH_FUNC__ engineFloat None(engineFloat x) { 
        return x; 
    }
    
    __MATH_FUNC__ engineFloat NoneDeriv(engineFloat x) { 
        return 1.0f; 
    }
    
    __MATH_FUNC__ engineFloat NoneSecondDeriv(engineFloat x) { 
        return 0.0f; 
    }
    
    // SIGMOID
    __MATH_FUNC__ engineFloat Sigmoid(engineFloat x) { 
        return 1.0f / (1.0f + std::exp(-x)); 
    }
    
    __MATH_FUNC__ engineFloat SigmoidDeriv(engineFloat x) { 
        engineFloat s = Sigmoid(x);
        return s * (1.0f - s);
    }
    
    __MATH_FUNC__ engineFloat SigmoidSecondDeriv(engineFloat x) { 
        engineFloat sig = Sigmoid(x);
        return sig * (1.0f - sig) * (1.0f - 2.0f * sig);
    }
    
    // RELU
    __MATH_FUNC__ engineFloat ReLU(engineFloat x) { 
        return std::max(static_cast<engineFloat>(0.0), x);
    }
    
    __MATH_FUNC__ engineFloat ReLUDeriv(engineFloat x) { 
        return x > 0.0f ? 1.0f : 0.0f; 
    }
    
    __MATH_FUNC__ engineFloat ReLUSecondDeriv(engineFloat x) { 
        return 0.0f; 
    }
    
    // LEAKY RELU
    __MATH_FUNC__ engineFloat LeakyReLU(engineFloat x) { 
        return x >= 0.0f ? x : x * Config::LeakyReLU_slope; 
    }
    
    __MATH_FUNC__ engineFloat LeakyReLUDeriv(engineFloat x) { 
        return x >= 0.0f ? 1.0f : Config::LeakyReLU_slope; 
    }
    
    __MATH_FUNC__ engineFloat LeakyReLUSecondDeriv(engineFloat x) { 
        return 0.0f; 
    }
    
    // TANH
    __MATH_FUNC__ engineFloat Tanh(engineFloat x) { 
        return std::tanh(x); 
    }
    
    __MATH_FUNC__ engineFloat TanhDeriv(engineFloat x) { 
        engineFloat t = std::tanh(x);
        return 1.0f - (t * t);
    }
    
    __MATH_FUNC__ engineFloat TanhSecondDeriv(engineFloat x) { 
        engineFloat t = std::tanh(x);
        return -2.0f * t * (1.0f - (t * t));
    }
    
    // SIREN
    __MATH_FUNC__ engineFloat Siren(engineFloat x) { 
        return std::sin(Config::Siren_w0 * x); 
    }
    
    __MATH_FUNC__ engineFloat SirenDeriv(engineFloat x) { 
        return Config::Siren_w0 * std::cos(Config::Siren_w0 * x);
    }
    
    __MATH_FUNC__ engineFloat SirenSecondDeriv(engineFloat x) { 
        return -1.0f * (Config::Siren_w0 * Config::Siren_w0) * std::sin(Config::Siren_w0 * x);
    }
    
    // FINER
    // sigma(x) = sin(w0 * (|x|+1) * x)
    // Same idea as SIREN, but the effective frequency grows with |x| instead of
    // being fixed at w0, so a single layer can access far more of the sine's
    // frequency range without changing weight scale. Frequency range is tuned by
    // the bias initialization (see GenerateInitialBiases) instead of weight scale.
    __MATH_FUNC__ engineFloat Finer(engineFloat x) {
        const engineFloat scale = std::abs(x) + 1.0f;
        return std::sin(Config::Finer_w0 * scale * x);
    }

    __MATH_FUNC__ engineFloat FinerDeriv(engineFloat x) {
        const engineFloat absX = std::abs(x);
        const engineFloat scale = absX + 1.0f;
        const engineFloat g = Config::Finer_w0 * scale * x;         // inner function
        const engineFloat gPrime = Config::Finer_w0 * (1.0f + 2.0f * absX); // d/dx of scale*x
        return std::cos(g) * gPrime;
    }

    __MATH_FUNC__ engineFloat FinerSecondDeriv(engineFloat x) {
        const engineFloat absX = std::abs(x);
        const engineFloat scale = absX + 1.0f;
        const engineFloat sign = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        const engineFloat g = Config::Finer_w0 * scale * x;
        const engineFloat gPrime = Config::Finer_w0 * (1.0f + 2.0f * absX);
        const engineFloat gDoublePrime = Config::Finer_w0 * 2.0f * sign;
        return (-std::sin(g) * gPrime * gPrime) + (std::cos(g) * gDoublePrime);
    }

    // WIRE (Dual-Weight)
    __MATH_FUNC__ engineFloat Wire(engineFloat zFreq, engineFloat zScale) {
        return std::exp(-Config::Wire_s_squared * (zScale * zScale)) * std::cos(Config::Wire_w0 * zFreq);
    }

    // Using pass-by-reference to avoid std::pair compatibility issues on the GPU
    __MATH_FUNC__ void WireDualDeriv(
        engineFloat zFreq, engineFloat zScale, 
        engineFloat& out_dFreq, engineFloat& out_dScale) 
    {
        engineFloat window = std::exp(-Config::Wire_s_squared * (zScale * zScale));
        engineFloat wave = std::cos(Config::Wire_w0 * zFreq);
        engineFloat out = window * wave;
        
        out_dFreq = window * (-Config::Wire_w0 * std::sin(Config::Wire_w0 * zFreq));
        out_dScale = -2.0f * Config::Wire_s_squared * zScale * out; 
    }

    // Using pass-by-reference to avoid std::tuple compatibility issues on the GPU
    __MATH_FUNC__ void WireDualSecondDeriv(
        engineFloat zFreq, engineFloat zScale, 
        engineFloat& out_d2Freq, engineFloat& out_d2Scale, engineFloat& out_d2Mixed) 
    {
        engineFloat window = std::exp(-Config::Wire_s_squared * (zScale * zScale));
        engineFloat wave = std::cos(Config::Wire_w0 * zFreq);
        engineFloat out = window * wave;
        
        out_d2Freq = -(Config::Wire_w0 * Config::Wire_w0) * out;
        out_d2Scale = window * wave * (4.0f * Config::Wire_s_squared * Config::Wire_s_squared * zScale * zScale - 2.0f * Config::Wire_s_squared);
        out_d2Mixed = -2.0f * Config::Wire_s_squared * zScale * window * (-Config::Wire_w0 * std::sin(Config::Wire_w0 * zFreq));
    }
}