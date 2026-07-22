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
		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut) const = 0;
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

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut) const override
		{
			return 0;
		}
	};

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

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut) const override
		{
			//// random value between -1 and 1 -- works but outdated
			return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
		
			// Xavier/Glorot better here: (but fanout is still wrong , todo)
			//float stddev = std::sqrt(2.0f / (fanIn + fanOut));
			//std::normal_distribution<float> distribution(0.0f, stddev);
			//return distribution(generator);
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

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut) const override
		{
			// He initialization
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

		virtual float GenerateInitialWeight(std::mt19937& generator, size_t fanIn, size_t fanOut) const override
		{
			// He initialization
			float stddev = std::sqrt(2.0f / fanIn);
			std::normal_distribution<float> distribution(0.0f, stddev);
			return distribution(generator);
		}

		virtual float GetLearningRateMultiplier() const override { return 0.01f; }
	};
}