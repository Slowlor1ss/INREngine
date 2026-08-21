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
    WireHybrid = 8,
};

// TODO: really we should make these functions __global__ or something and replace out cpu code to also use these for when running with --use_gpu off
// right now i did a more or less lazy solution of calling the functions in ActivationFunctions.h but this could all be a lot cleaner
namespace SharedAct 
{
    // Hyperparameters
    namespace Config 
    {
        constexpr engineFloat Siren_w0 = 30.0f;
        
        constexpr engineFloat Wire_w0 = 4.0f;//20.0f;
        constexpr engineFloat Wire_s = 4.0f;//30.0f;
        constexpr engineFloat Wire_s_squared = Wire_s * Wire_s;
        
        constexpr engineFloat LeakyReLU_slope = 0.1f;

        // FINER (variable-periodic sine): sin(w0 * (|x|+1) * x)
        constexpr engineFloat Finer_w0 = 5.f; // 30.0f;
        // Bias init range for FINER, this is what actually gives it its extra
        // frequency range over SIREN, see GenerateInitialBiases below
        constexpr engineFloat Finer_bias_k = 0.5f;//2.5f;

        constexpr engineFloat Hybrid_mix = 0.20f;
		constexpr engineFloat Hybrid_frequency_multiplier = 1.95f;
        constexpr engineFloat Hybrid_mix2 = 0.05f;
		constexpr engineFloat Hybrid_frequency_multiplier2 = 2.9f;
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

	// WIRE + envelope-controlled high-frequency residual
	__MATH_FUNC__ engineFloat WireHybrid(
	    engineFloat zFreq,
	    engineFloat zScale)
	{
	    engineFloat window =
	        std::exp(
	            -Config::Wire_s_squared *
	            (zScale * zScale)
	        );

	    engineFloat wire =
	        window *
	        std::cos(
	            Config::Wire_w0 * zFreq
	        );


	    // --------------------------------------------------------
	    // First high-frequency residual
	    // --------------------------------------------------------

	    engineFloat sharp1 =
	        std::sin(
	            Config::Wire_w0 *
	            Config::Hybrid_frequency_multiplier *
	            zFreq
	        );


	    // --------------------------------------------------------
	    // Second high-frequency residual
	    // --------------------------------------------------------

	    engineFloat sharp2 =
	        std::sin(
	            Config::Wire_w0 *
	            Config::Hybrid_frequency_multiplier2 *
	            zFreq
	        );


	    return wire +
	        Config::Hybrid_mix * sharp1 +
	        Config::Hybrid_mix2 * sharp2;
	}

    // Using pass-by-reference to avoid std::pair compatibility issues on the GPU
	__MATH_FUNC__ void WireHybridDualDeriv(
    engineFloat zFreq,
    engineFloat zScale,
    engineFloat& out_dFreq,
    engineFloat& out_dScale)
	{
	    engineFloat s2 =
	        Config::Wire_s_squared;

	    engineFloat w0 =
	        Config::Wire_w0;

	    engineFloat mix1 =
	        Config::Hybrid_mix;

	    engineFloat freqMult1 =
	        Config::Hybrid_frequency_multiplier;

	    engineFloat mix2 =
	        Config::Hybrid_mix2;

	    engineFloat freqMult2 =
	        Config::Hybrid_frequency_multiplier2;


	    // --------------------------------------------------------
	    // WIRE
	    // --------------------------------------------------------

	    engineFloat window =
	        std::exp(
	            -s2 * (zScale * zScale)
	        );

	    engineFloat wireWave =
	        std::cos(
	            w0 * zFreq
	        );

	    engineFloat wire =
	        window * wireWave;


	    // --------------------------------------------------------
	    // First residual
	    // --------------------------------------------------------

	    engineFloat argument1 =
	        w0 * freqMult1 * zFreq;

	    engineFloat sharp1 =
	        std::sin(argument1);


	    // --------------------------------------------------------
	    // Second residual
	    // --------------------------------------------------------

	    engineFloat argument2 =
	        w0 * freqMult2 * zFreq;

	    engineFloat sharp2 =
	        std::sin(argument2);


	    // --------------------------------------------------------
	    // d/dzFreq
	    // --------------------------------------------------------

	    engineFloat wire_dFreq =
	        window *
	        (
	            -w0 *
	            std::sin(w0 * zFreq)
	        );

	    engineFloat sharp1_dFreq =
	        w0 *
	        freqMult1 *
	        std::cos(argument1);

	    engineFloat sharp2_dFreq =
	        w0 *
	        freqMult2 *
	        std::cos(argument2);

	    out_dFreq =
	        wire_dFreq +
	        mix1 * sharp1_dFreq +
	        mix2 * sharp2_dFreq;
        // --------------------------------------------------------
	    // d/dzScale
	    // --------------------------------------------------------

	    out_dScale =
	        -2.0f *
	        s2 *
	        zScale *
	        wire;
	}

    // Using pass-by-reference to avoid std::tuple compatibility issues on the GPU
	__MATH_FUNC__ void WireHybridDualSecondDeriv(
	    engineFloat zFreq,
	    engineFloat zScale,
	    engineFloat& out_d2Freq,
	    engineFloat& out_d2Scale,
	    engineFloat& out_d2Mixed)
	{
	    engineFloat s2 =
	        Config::Wire_s_squared;

	    engineFloat w0 =
	        Config::Wire_w0;

	    engineFloat mix1 =
	        Config::Hybrid_mix;

	    engineFloat freqMult1 =
	        Config::Hybrid_frequency_multiplier;

	    engineFloat mix2 =
	        Config::Hybrid_mix2;

	    engineFloat freqMult2 =
	        Config::Hybrid_frequency_multiplier2;

	// --------------------------------------------------------
	    // WIRE
	    // --------------------------------------------------------

	    engineFloat window =
	        std::exp(
	            -s2 * (zScale * zScale)
	        );

	    engineFloat wave =
	        std::cos(
	            w0 * zFreq
	        );

	    engineFloat wire =
	        window * wave;

	    // --------------------------------------------------------
	    // First residual
	    // --------------------------------------------------------

	    engineFloat argument1 =
	        w0 * freqMult1 * zFreq;

	    engineFloat sharp1 =
	        std::sin(argument1);


	    // --------------------------------------------------------
	    // Second residual
	    // --------------------------------------------------------

	    engineFloat argument2 =
	        w0 * freqMult2 * zFreq;

	    engineFloat sharp2 =
	        std::sin(argument2);


	    // --------------------------------------------------------
	    // d2/dzFreq2
	    // --------------------------------------------------------

	    engineFloat wire_d2Freq =
	        -(w0 * w0) *
	        wire;

	    engineFloat sharp1_d2Freq =
	        -(w0 * w0 *
	          freqMult1 * freqMult1) *
	        sharp1;

	    engineFloat sharp2_d2Freq =
	        -(w0 * w0 *
	          freqMult2 * freqMult2) *
	        sharp2;

	    out_d2Freq =
	        wire_d2Freq +
	        mix1 * sharp1_d2Freq +
	        mix2 * sharp2_d2Freq;


	    // --------------------------------------------------------
	    // d2/dzScale2
	    // --------------------------------------------------------

	    out_d2Scale =
	        window *
	        wave *
	        (
	            4.0f *
	            s2 * s2 *
	            zScale * zScale
	            -
	            2.0f * s2
	        );


	    // --------------------------------------------------------
	    // Mixed derivative
	    // --------------------------------------------------------

	    out_d2Mixed =
	        -2.0f *
	        s2 *
	        zScale *
	        window *
	        (
	            -w0 *
	            std::sin(w0 * zFreq)
	        );
	}
    
    //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    // GPU KERNEL FORWARD ROUTER
    __MATH_FUNC__ engineFloat Execute(GpuActType actType, engineFloat zFreq, engineFloat zScale)
    {
        switch (actType) {
        case GpuActType::Sigmoid:   return Sigmoid(zFreq);
        case GpuActType::ReLU:      return ReLU(zFreq);
        case GpuActType::LeakyReLU: return LeakyReLU(zFreq);
        case GpuActType::Siren:     return Siren(zFreq);
        case GpuActType::Wire:      return Wire(zFreq, zScale);
        case GpuActType::Tanh:      return Tanh(zFreq);
        case GpuActType::Finer:     return Finer(zFreq);
        case GpuActType::WireHybrid: return WireHybrid(zFreq, zScale);
        case GpuActType::None:
        default:                    return None(zFreq);
        }
    }
    
    // GPU KERNEL DERIVATIVE ROUTER
    __MATH_FUNC__ void ExecuteDerivatives(
        GpuActType actType, engineFloat zFreq, engineFloat zScale,
        engineFloat& deriv1Freq, engineFloat& deriv1Scale,
        engineFloat& deriv2Freq, engineFloat& deriv2Scale, engineFloat& deriv2Mixed)
    {
        // Default fallbacks
        deriv1Freq = 1.0f; deriv1Scale = 0.0f;
        deriv2Freq = 0.0f; deriv2Scale = 0.0f; deriv2Mixed = 0.0f;

        switch (actType) {
        case GpuActType::Sigmoid:
            deriv1Freq = SigmoidDeriv(zFreq);
            deriv2Freq = SigmoidSecondDeriv(zFreq);
            break;
        case GpuActType::ReLU:
            deriv1Freq = ReLUDeriv(zFreq);
            deriv2Freq = ReLUSecondDeriv(zFreq);
            break;
        case GpuActType::LeakyReLU:
            deriv1Freq = LeakyReLUDeriv(zFreq);
            deriv2Freq = LeakyReLUSecondDeriv(zFreq);
            break;
        case GpuActType::Siren:
            deriv1Freq = SirenDeriv(zFreq);
            deriv2Freq = SirenSecondDeriv(zFreq);
            break;
        case GpuActType::Tanh:
            deriv1Freq = TanhDeriv(zFreq);
            deriv2Freq = TanhSecondDeriv(zFreq);
            break;
        case GpuActType::Wire:
            WireDualDeriv(zFreq, zScale, deriv1Freq, deriv1Scale);
            WireDualSecondDeriv(zFreq, zScale, deriv2Freq, deriv2Scale, deriv2Mixed);
            break;
        case GpuActType::Finer:
            deriv1Freq = FinerDeriv(zFreq);
            deriv2Freq = FinerSecondDeriv(zFreq);
            break;
        case GpuActType::WireHybrid:
            WireHybridDualDeriv(zFreq, zScale, deriv1Freq, deriv1Scale);
            WireHybridDualSecondDeriv(zFreq, zScale, deriv2Freq, deriv2Scale, deriv2Mixed);
            break;
        case GpuActType::None:
        default:
            deriv1Freq = NoneDeriv(zFreq);
            deriv2Freq = NoneSecondDeriv(zFreq);
            break;
        }
    }
}