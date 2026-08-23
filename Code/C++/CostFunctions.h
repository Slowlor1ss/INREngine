#pragma once
#include <string>
#include <cmath>
#include <algorithm>

#include "../cuda/GpuCostFunctions.cuh"
#include "Types.h"

namespace CostFunc
{
	class Base
	{
	public:
		virtual ~Base() = default;
		virtual std::string GetName() const = 0;
		virtual GpuCostType GetGpuType() const = 0;
		virtual engineFloat Execute(engineFloat activation, engineFloat target) const = 0;
		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const = 0;
	};

	class MSE : public Base
	{
	public:
		static constexpr const char* k_name{ "MSE" };
		virtual GpuCostType GetGpuType() const override { return GpuCostType::MSE; }

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat activation, engineFloat target) const override
		{
			return SharedCost::MSE(activation, target);
		}

		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const override
		{
			// derivative of (act - y)^2 is 2 * (act - y)
			return SharedCost::MSEDeriv(activation, target);
		}
	};

	class L1 : public Base
	{
	public:
		static constexpr const char* k_name{ "L1" };
		virtual GpuCostType GetGpuType() const override { return GpuCostType::L1; }

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat activation, engineFloat target) const override
		{
			return SharedCost::L1(activation, target);
		}

		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const override
		{
			// derivative of l1 loss: -1 if target > activation, +1 if activation >= target
			return SharedCost::L1Deriv(activation, target);
		}
	};
	
	class Charbonnier : public Base
	{
	public:
		static constexpr const char* k_name{ "Charbonnier" };
		virtual GpuCostType GetGpuType() const override { return GpuCostType::Charbonnier; }
		// Small epsilon in normalized [0,1] pixel space, big enough to keep the
		// gradient well-behaved near zero error, small enough to stay close to L1
		//static constexpr engineFloat k_epsilon{ 1e-3f };
		//static constexpr engineFloat k_epsilonSquared{ k_epsilon * k_epsilon };

		virtual std::string GetName() const override
		{
			return k_name;
		}

		virtual engineFloat Execute(engineFloat activation, engineFloat target) const override
		{
			return SharedCost::Charbonnier(activation, target);
		}

		virtual engineFloat ExecuteDerivative(engineFloat activation, engineFloat target) const override
		{
			// d/d(activation) sqrt(diff^2 + eps^2) = diff / sqrt(diff^2 + eps^2)
			return SharedCost::CharbonnierDeriv(activation, target);
		}
	};
}