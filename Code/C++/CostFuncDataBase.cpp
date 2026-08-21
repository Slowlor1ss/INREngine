#include "CostFuncDataBase.h"

namespace CostFunc
{
	DataBase* DataBase::m_instance = nullptr;

	DataBase* DataBase::GetOrCreateInstance()
	{
		static DataBase instance{};
		if (m_instance == nullptr)
		{
			m_instance = &instance;
			instance.m_map.emplace(MSE::k_name, std::make_unique<MSE>());
			instance.m_map.emplace(L1::k_name, std::make_unique<L1>());
			instance.m_map.emplace(Charbonnier::k_name, std::make_unique<Charbonnier>());
		}

		return m_instance;
	}

	Base* DataBase::FindCostFunc(const std::string& name)
	{
		return DataBase::GetOrCreateInstance()->m_map.at(name).get();
	}
}
