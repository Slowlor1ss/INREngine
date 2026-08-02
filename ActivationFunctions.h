#pragma once
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

namespace ActFunc
{
	class Base
	{
	public:
		virtual std::string GetName() const = 0;
		virtual float Execute(float x) const = 0;
		virtual float ExecuteDerivative(float x) const = 0;
		virtual float ExecuteSecondDerivative(float x) const = 0;
		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const = 0;
		virtual float GetLearningRateMultiplier() const { return 1.0f; }
	};

	class None : public Base
	{
	public:

		static constexpr const char* k_name{ "None" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual float Execute(float x) const override
		{
			return x;
		}

		virtual float ExecuteDerivative(float x) const override
		{
			return 1;
		}
		
		virtual float ExecuteSecondDerivative(float x) const override
		{
			// The first derivative of f(x)=x is 1.0. 
			// The derivative of a constant 1.0 is 0.0.
			return 0.0f; 
		}

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// Standard Xavier initialization
			const float stddev = std::sqrt(2.0f / static_cast<float>(fanIn + fanOut));
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
	// 	virtual float Execute(float x) const override
	// 	{
	// 		// Squash numbers into a 0.0 to 1.0 range
	// 		return 0.5f * (x / (1.0f + std::abs(x)) + 1.0f);
	// 	}
	//
	// 	virtual float ExecuteDerivative(float x) const override
	// 	{
	// 		const float denom = 1.0f + std::abs(x);
	// 		return 0.5f / (denom * denom);
	// 	}
	//
	// 	virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
	// 	{
	// 		// Standard Xavier initialization
	// 		const float stddev = std::sqrt(2.0f / static_cast<float>(fanIn + fanOut));
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

		virtual float Execute(float x) const override
		{
			return 1.0f / (1.0f + expf(-x));
		}

		virtual float ExecuteDerivative(float x) const override
		{
			float s = Execute(x);
			return s * (1.0f - s);
		}
		
		virtual float ExecuteSecondDerivative(float x) const override
		{
			float sig = 1.0f / (1.0f + std::exp(-x));
			return sig * (1.0f - sig) * (1.0f - 2.0f * sig);
		}

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// Xavier initialization bounds
		    float bound = std::sqrt(6.0f / static_cast<float>(fanIn));
		    std::uniform_real_distribution<float> distribution(-bound, bound);
		    return distribution(generator);
		}

	};

	class ReLU : public Base
	{
	public:

		static constexpr const char* k_name{ "ReLU" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual float Execute(float x) const override
		{
			return std::max(0.0f, x);
		}

		virtual float ExecuteDerivative(float x) const override
		{
			return x > 0.0f ? 1.0f : 0.0f;
		}
		
		virtual float ExecuteSecondDerivative(float x) const override
		{
			return 0.0f;
		}

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// He initialization // Note-LKrikilion: This is so graphics programmer of you; what the hell is "He"?
			float stddev = std::sqrt(2.0f / fanIn);
			std::normal_distribution<float> distribution(0.0f, stddev);
			return distribution(generator);
		}

		virtual float GetLearningRateMultiplier() const override { return 0.01f; }
	};

	class LeakyReLU : public Base
	{
		static constexpr float k_leakySlope = 0.1f;
	public:

		static constexpr const char* k_name{ "LeakyReLU" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual float Execute(float x) const override
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

		virtual float ExecuteDerivative(float x) const override
		{
			return x >= 0.0f ? 1.0f : k_leakySlope;
		}
		
		virtual float ExecuteSecondDerivative(float x) const override
		{
			return 0.0f;
		}

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// He initialization
			float stddev = std::sqrt(2.0f / fanIn);
			std::normal_distribution<float> distribution(0.0f, stddev);
			return distribution(generator);
		}

		virtual float GetLearningRateMultiplier() const override { return 0.01f; }
	};

	// Heavily based on https://deepwiki.com/vsitzmann/siren
	class Siren : public Base
	{
	public:
		static constexpr const char* k_name{ "Siren" };
		static constexpr float k_w0 = 30.0f; // SIREN frequency hyperparameter (omega_naught)

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual float Execute(float x) const override
		{
			return std::sin(k_w0 * x);
		}

		virtual float ExecuteDerivative(float x) const override
		{
			return k_w0 * std::cos(k_w0 * x);
		}

		virtual float ExecuteSecondDerivative(float x) const override
		{
		    // Second derivative of sin(w0 * x)
		    return -1.0f * (k_w0 * k_w0) * std::sin(k_w0 * x);
		}

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// SIREN-specific initialization:
			// First layer: Weights are initialized uniformly between -1/in_features and 1/in_features
			// Subsequent layers: Weights are initialized uniformly between -sqrt(6/in_features)/w0 and sqrt(6/in_features)/w0
			// https://deepwiki.com/vsitzmann/siren#key-properties-of-siren
			const float bound = (layerIndex == 1
				                    ? 1.0f / static_cast<float>(fanIn)
				                    : std::sqrt(6.0f / static_cast<float>(fanIn)) / k_w0);

			std::uniform_real_distribution distribution(-bound, bound);
			return distribution(generator);
		}

		// TODO-LKrikilion: mess around with this value a bit on a better machine 
		virtual float GetLearningRateMultiplier() const override { return 1.0f; }//0.0001f; }
	};
	
	// For debugging
	class Tanh : public Base
	{
	public:
		static constexpr const char* k_name{ "Tanh" };
		virtual std::string GetName() const override { return k_name; }

		virtual float Execute(float x) const override
		{
			return std::tanh(x);
		}

		virtual float ExecuteDerivative(float x) const override
		{
			float t = std::tanh(x);
			// f'(x) = 1 - tanh(x)^2
			return 1.0f - (t * t);
		}

		virtual float ExecuteSecondDerivative(float x) const override
		{
			float t = std::tanh(x);
			// f''(x) = -2 * tanh(x) * (1 - tanh(x)^2)
			return -2.0f * t * (1.0f - (t * t));
		}

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut, size_t layerIndex) const override
		{
			// Xavier/Glorot Initialization (perfectly balances Tanh networks)
			float limit = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
			std::uniform_real_distribution<float> dist(-limit, limit);
			return dist(generator);
		}
	};
}