#pragma once
#include <string>
#include <cmath>
#include <algorithm>

#include "Types.h"

namespace CostFunc
{
	class Base
	{
	public:
		virtual ~Base() = default;
		virtual std::string GetName() const = 0;
		virtual engineFloat Execute(engineFloat activation, engineFloat target) const = 0;
		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const = 0;
	};

	class MSE : public Base
	{
	public:
		static constexpr const char* k_name{ "MSE" };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat activation, engineFloat target) const override
		{
			engineFloat diff = activation - target;
			return diff * diff;
		}

		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const override
		{
			// derivative of (act - y)^2 is 2 * (act - y)
			return 2.0f * (activation - target);
		}
	};

	class Charbonnier : public Base
	{
	public:
		static constexpr const char* k_name{ "Charbonnier" };
		// Small epsilon in normalized [0,1] pixel space -- big enough to keep the
		// gradient well-behaved near zero error, small enough to stay close to L1.
		static constexpr engineFloat k_epsilon{ 1e-3f };
		static constexpr engineFloat k_epsilonSquared{ k_epsilon * k_epsilon };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat activation, engineFloat target) const override
		{
			engineFloat diff = activation - target;
			return std::sqrt(diff * diff + k_epsilonSquared);
		}

		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const override
		{
			engineFloat diff = activation - target;
			// d/d(activation) sqrt(diff^2 + eps^2) = diff / sqrt(diff^2 + eps^2)
			return diff / std::sqrt(diff * diff + k_epsilonSquared);
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

		virtual engineFloat Execute(engineFloat activation, engineFloat target) const override
		{
			return std::abs(activation - target);
		}

		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const override
		{
			// derivative of l1 loss: -1 if target > activation, +1 if activation >= target
			return (target > activation) ? -1.0f : 1.0f;
		}
	};
}