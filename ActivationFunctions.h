#pragma once
#include <string>
#include <cmath>
#include <algorithm>

namespace ActFunc
{
	class Base
	{
	public:
		virtual std::string GetName() const = 0;
		virtual float Execute(float x) const = 0;
		virtual float ExecuteDerivative(float x) const = 0;
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
	};
}