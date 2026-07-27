#pragma once
#include <string>
#include <cmath>
#include <algorithm>

namespace CostFunc
{
	class Base
	{
	public:
		virtual ~Base() = default;
		virtual std::string GetName() const = 0;
		virtual float Execute(float activation, float target) const = 0;
		virtual float ExecuteDerivative(float activation, float target) const = 0;
	};

	class MSE : public Base
	{
	public:
		static constexpr const char* k_name{ "MSE" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual float Execute(float activation, float target) const override
		{
			float diff = activation - target;
			return diff * diff;
		}

		virtual float ExecuteDerivative(float activation, float target) const override
		{
			// derivative of (act - y)^2 is 2 * (act - y)
			return 2.0f * (activation - target);
		}
	};

	class L1 : public Base
	{
	public:
		static constexpr const char* k_name{ "L1" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual float Execute(float activation, float target) const override
		{
			return std::abs(activation - target);
		}

		virtual float ExecuteDerivative(float activation, float target) const override
		{
			// derivative of l1 loss: -1 if target > activation, +1 if activation >= target
			return (target > activation) ? -1.0f : 1.0f;
		}
	};
}
