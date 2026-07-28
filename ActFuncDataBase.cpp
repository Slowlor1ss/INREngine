#include "ActFuncDataBase.h"

namespace ActFunc
{
	DataBase* DataBase::m_instance = nullptr;

	DataBase* DataBase::GetOrCreateInstance()
	{
		static DataBase instance{};
		if (m_instance == nullptr)
		{
			m_instance = &instance;
			instance.m_map.emplace(Empty::k_name, std::make_unique<Empty>());
			instance.m_map.emplace(Sigmoid::k_name, std::make_unique<Sigmoid>());
			instance.m_map.emplace(ReLU::k_name, std::make_unique<ReLU>());
			instance.m_map.emplace(LeakyReLU::k_name, std::make_unique<LeakyReLU>());
			instance.m_map.emplace(None::k_name, std::make_unique<None>());

		}

		return m_instance;
	}

	Base* DataBase::FindActFunc(const std::string& name)
	{
		return DataBase::GetOrCreateInstance()->m_map.at(name).get();
	}
}