#pragma once
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

#include "Types.h"

namespace ActFunc
{
	class Base
	{
	public:
		virtual std::string GetName() const = 0;
		
		virtual engineFloat Execute(engineFloat x) const = 0;
		virtual engineFloat ExecuteDerivative(engineFloat x) const = 0;
		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const = 0;
		
		// Dual-Weight Forward Pass
		// Default fallback: Ignore the scale Z and just run the standard Execute
		virtual engineFloat ExecuteDual(engineFloat zFreq, engineFloat zScale) const 
		{
			return Execute(zFreq); 
		}
		// Dual-Weight Backward Pass
		// Returns a pair: { Derivative_wrt_Freq, Derivative_wrt_Scale }
		// Default fallback: Return the standard derivative for freq, and 0.0 for scale
		virtual std::pair<engineFloat, engineFloat> ExecuteDualDerivative(engineFloat zFreq, engineFloat zScale) const
		{
			return { ExecuteDerivative(zFreq), 0.f };
		}
		virtual std::tuple<engineFloat, engineFloat, engineFloat> ExecuteDualSecondDerivative(engineFloat zFreq, engineFloat zScale) const
		{
			return { ExecuteSecondDerivative(zFreq), 0.f, 0.f };
		}
		
		virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const = 0;
		// Generally biases are safer to initialize as straight 0
		virtual engineFloat GenerateInitialBiases(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) { return 0.0f; }
		
		virtual engineFloat GetLearningRateMultiplier() const { return 1.0f; }
	};

	class None : public Base
	{
	public:

		static constexpr const char* k_name{ "None" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat x) const override
		{
			return x;
		}

		virtual engineFloat ExecuteDerivative(engineFloat x) const override
		{
			return 1;
		}
		
		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const override
		{
			// The first derivative of f(x)=x is 1.0. 
			// The derivative of a constant 1.0 is 0.0.
			return 0.0f; 
		}

		virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// Standard Xavier initialization
			const engineFloat stddev = std::sqrt(2.0f / static_cast<engineFloat>(fanIn + fanOut));
			std::uniform_real_distribution distribution(-stddev, stddev);
			return distribution(generator);
		}
	};

	// Purely made to keep the colours withing a valid range so the representation looks better in the early stages
	// we could also use some of the other activator functions but that just unnessesarily expensive
	// class ColorSquash : public Base
	// {
	// public:
	// 	static constexpr const char* k_name{ "FastColorSquash" };
	//
	// 	virtual std::string GetName() const override
	// 	{
	// 		return k_name;
	// 	}
	//
	// 	virtual engineFloat Execute(engineFloat x) const override
	// 	{
	// 		// Squash numbers into a 0.0 to 1.0 range
	// 		return 0.5f * (x / (1.0f + std::abs(x)) + 1.0f);
	// 	}
	//
	// 	virtual engineFloat ExecuteDerivative(engineFloat x) const override
	// 	{
	// 		const engineFloat denom = 1.0f + std::abs(x);
	// 		return 0.5f / (denom * denom);
	// 	}
	//
	// 	virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
	// 	{
	// 		// Standard Xavier initialization
	// 		const engineFloat stddev = std::sqrt(2.0f / static_cast<engineFloat>(fanIn + fanOut));
	// 		std::uniform_real_distribution distribution(-stddev, stddev);
	// 		return distribution(generator);
	// 	}
	// };

	class Sigmoid : public Base
	{
	public:

		static constexpr const char* k_name{ "Sigmoid" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat x) const override
		{
			return 1.0f / (1.0f + std::exp(-x));
		}

		virtual engineFloat ExecuteDerivative(engineFloat x) const override
		{
			engineFloat s = Execute(x);
			return s * (1.0f - s);
		}
		
		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const override
		{
			engineFloat sig = 1.0f / (1.0f + std::exp(-x));
			return sig * (1.0f - sig) * (1.0f - 2.0f * sig);
		}

		virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// Xavier initialization bounds
		    engineFloat bound = std::sqrt(6.0f / static_cast<engineFloat>(fanIn));
		    std::uniform_real_distribution<engineFloat> distribution(-bound, bound);
		    return distribution(generator);
		}

		virtual engineFloat GetLearningRateMultiplier() const override { return 0.001f; }
	};

	class ReLU : public Base
	{
	public:

		static constexpr const char* k_name{ "ReLU" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat x) const override
		{
			return std::max((engineFloat)0, x);
		}

		virtual engineFloat ExecuteDerivative(engineFloat x) const override
		{
			return x > 0.0f ? 1.0f : 0.0f;
		}
		
		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const override
		{
			return 0.0f;
		}

		virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// He initialization // Note-LKrikilion: This is so graphics programmer of you; what the hell is "He"?
			engineFloat stddev = std::sqrt(2 / (engineFloat)fanIn);
			std::normal_distribution<engineFloat> distribution(0.0f, stddev);
			return distribution(generator);
		}

		virtual engineFloat GetLearningRateMultiplier() const override { return 0.0005f; }
	};

	class LeakyReLU : public Base
	{
		static constexpr engineFloat k_leakySlope = 0.1f;
	public:

		static constexpr const char* k_name{ "LeakyReLU" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat x) const override
		{
			if (x >= 0.0f)
			{
				return x;
			}
			else
			{
				return x * k_leakySlope;
			}
		}

		virtual engineFloat ExecuteDerivative(engineFloat x) const override
		{
			return x >= 0.0f ? 1.0f : k_leakySlope;
		}
		
		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const override
		{
			return 0.0f;
		}

		virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// He initialization
			engineFloat stddev = std::sqrt(2.0f / (engineFloat)fanIn);
			std::normal_distribution<engineFloat> distribution(0.0f, stddev);
			return distribution(generator);
		}

		virtual engineFloat GetLearningRateMultiplier() const override { return 0.0005f; }
	};

	// Heavily based on https://deepwiki.com/vsitzmann/siren
	class Siren : public Base
	{
	public:
		static constexpr const char* k_name{ "Siren" };
		static constexpr engineFloat k_w0 = 30.f;//30.0f; // SIREN frequency hyperparameter (omega_naught)

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat x) const override
		{
			return std::sin(k_w0 * x);
		}

		virtual engineFloat ExecuteDerivative(engineFloat x) const override
		{
			return k_w0 * std::cos(k_w0 * x);
		}

		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const override
		{
		    // Second derivative of sin(w0 * x)
		    return -1.0f * (k_w0 * k_w0) * std::sin(k_w0 * x);
		}

		virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// SIREN-specific initialization:
			// First layer: Weights are initialized uniformly between -1/in_features and 1/in_features
			// Subsequent layers: Weights are initialized uniformly between -sqrt(6/in_features)/w0 and sqrt(6/in_features)/w0
			// https://deepwiki.com/vsitzmann/siren#key-properties-of-siren
			const engineFloat bound = (layerIndex == 1
				                    ? 1.0f / static_cast<engineFloat>(fanIn)
				                    : std::sqrt(6.0f / static_cast<engineFloat>(fanIn)) / k_w0);

			std::uniform_real_distribution distribution(-bound, bound);
			return distribution(generator);
		}

		// TODO-LKrikilion: mess around with this value a bit on a better machine 
		virtual engineFloat GetLearningRateMultiplier() const override { return 0.0001f; }//0.0001f; }
	};
	
	// For debugging
	class Tanh : public Base
	{
	public:
		static constexpr const char* k_name{ "Tanh" };
		virtual std::string GetName() const override { return k_name; }

		virtual engineFloat Execute(engineFloat x) const override
		{
			return std::tanh(x);
		}

		virtual engineFloat ExecuteDerivative(engineFloat x) const override
		{
			engineFloat t = std::tanh(x);
			// f'(x) = 1 - tanh(x)^2
			return 1.0f - (t * t);
		}

		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const override
		{
			engineFloat t = std::tanh(x);
			// f''(x) = -2 * tanh(x) * (1 - tanh(x)^2)
			return -2.0f * t * (1.0f - (t * t));
		}

		virtual engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// Xavier/Glorot Initialization (perfectly balances Tanh networks)
			engineFloat limit = std::sqrt(6.0f / static_cast<engineFloat>(fanIn + fanOut));
			std::uniform_real_distribution<engineFloat> dist(-limit, limit);
			return dist(generator);
		}
		
		virtual engineFloat GetLearningRateMultiplier() const override { return 0.001f; }
	};
	
	// https://vishwa91.github.io/wire
	// Wavelet Implicit Neural Representations
	class Wire : public Base
	{
	public:
		static constexpr const char* k_name{ "Wire" };
    
		// Hyperparameters from the WIRE paper & python files
		// https://github.com/vishwa91/wire/blob/main/wire_image_denoise.py
		// We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
		static constexpr engineFloat k_w0 = 20.f;//30.f;//16.f;//20.0f; // Frequency (controls sharpness)
		static constexpr engineFloat k_s = 30.f;//45.f;//24.f;//30.0f;  // Scale (controls localization/smoothness)

		virtual std::string GetName() const override { return k_name; }
		
		// Standard pass (Used for the very first layer which doesn't have dual weights)
		engineFloat Execute(engineFloat z) const override
		{
			return ExecuteDual(z, z);
		}

		engineFloat ExecuteDerivative(engineFloat z) const override
		{
			return ExecuteDualDerivative(z, z).first; // Just grab the freq derivative
		}
		
		virtual engineFloat ExecuteSecondDerivative(engineFloat x) const override
		{
			engineFloat s2 = k_s * k_s;
			engineFloat E = std::exp(-s2 * x * x);
			engineFloat S = std::sin(k_w0 * x);
			engineFloat C = std::cos(k_w0 * x);

			engineFloat term1 = (4.0f * s2 * s2 * x * x) - (2.0f * s2) - (k_w0 * k_w0);
			engineFloat term2 = 4.0f * s2 * k_w0 * x;

			return E * (term1 * C + term2 * S);
		}

		// Dual weight pass
		engineFloat ExecuteDual(engineFloat zFreq, engineFloat zScale) const override
		{
			engineFloat s_squared = k_s * k_s;
			engineFloat window = std::exp(-s_squared * (zScale * zScale));
			engineFloat wave = std::cos(k_w0 * zFreq);
        
			return window * wave;
		}

		std::pair<engineFloat, engineFloat> ExecuteDualDerivative(engineFloat zFreq, engineFloat zScale) const override
		{
			engineFloat s_squared = k_s * k_s;
			engineFloat window = std::exp(-s_squared * (zScale * zScale));
			engineFloat wave = std::cos(k_w0 * zFreq);
			engineFloat out = window * wave;
        
			// Derivative with respect to the frequency Z
			engineFloat d_freq = window * (-k_w0 * std::sin(k_w0 * zFreq));
        
			// Derivative with respect to the scale (envelope) Z
			engineFloat d_scale = -2.0f * s_squared * zScale * out; 

			return { d_freq, d_scale };
		}
		
		// Im very unsure on my math on this one... but were not using it at the moment so TODO: later :D
		std::tuple<engineFloat, engineFloat, engineFloat> ExecuteDualSecondDerivative(engineFloat zFreq, engineFloat zScale) const override
		{
			engineFloat s_squared = k_s * k_s;
			engineFloat window = std::exp(-s_squared * (zScale * zScale));
			engineFloat wave = std::cos(k_w0 * zFreq);
			engineFloat out = window * wave;
        
			// Second derivative w.r.t Frequency (zFreq, zFreq)
			engineFloat d2_freq = - (k_w0 * k_w0) * out;

			// Second derivative w.r.t Scale (zScale, zScale)
			engineFloat d2_scale = window * wave * (4.0f * s_squared * s_squared * zScale * zScale - 2.0f * s_squared);

			// Mixed partial derivative (zFreq, zScale)
			engineFloat d2_mixed = -2.0f * s_squared * zScale * window * (-k_w0 * std::sin(k_w0 * zFreq));

			return { d2_freq, d2_scale, d2_mixed };
		}

		engineFloat GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// https://github.com/vishwa91/wire/blob/main/modules/wire.py
			// PyTorch default nn.Linear initialization: U(-sqrt(1/fan_in), sqrt(1/fan_in))
			const engineFloat bound = static_cast<engineFloat>(1.0 / std::sqrt(static_cast<double>(fanIn)));
			std::uniform_real_distribution<engineFloat> distribution(-bound, bound);
			return distribution(generator);
		}
		
		engineFloat GenerateInitialBiases(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) override
		{
			// We initialise our biases in the same way as we do out weights for Wire
			const engineFloat bound = 1.0f / std::sqrt(static_cast<engineFloat>(fanIn));
			std::uniform_real_distribution<engineFloat> distribution(-bound, bound);
			return distribution(generator);
		}
			
		// TODO: when we start using the LR multiplyer again
		// # WIRE works best at 5e-3 to 2e-2 (Source: https://github.com/vishwa91/wire/blob/main/wire_image_denoise.py)
		virtual engineFloat GetLearningRateMultiplier() const override { return 0.005f; }
	};
}