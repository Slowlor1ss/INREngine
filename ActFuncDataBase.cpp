#include "ActFuncDataBase.h"

#include <ranges>

namespace ActFunc
{
	DataBase* DataBase::m_instance = nullptr;

	DataBase* DataBase::GetOrCreateInstance()
	{
		static DataBase instance{};
		if (m_instance == nullptr)
		{
			m_instance = &instance;
			instance.m_map.emplace(None::k_name, std::make_unique<None>());
			//instance.m_map.emplace(ColorSquash::k_name, std::make_unique<ColorSquash>());
			instance.m_map.emplace(Sigmoid::k_name, std::make_unique<Sigmoid>());
			instance.m_map.emplace(ReLU::k_name, std::make_unique<ReLU>());
			instance.m_map.emplace(LeakyReLU::k_name, std::make_unique<LeakyReLU>());
			instance.m_map.emplace(Siren::k_name, std::make_unique<Siren>());
			instance.m_map.emplace(Tanh::k_name, std::make_unique<Tanh>());
			instance.m_map.emplace(Wire::k_name, std::make_unique<Wire>());
			instance.m_map.emplace(Finer::k_name, std::make_unique<Finer>());
		}

		return m_instance;
	}

	Base* DataBase::FindActFunc(const std::string& name)
	{
		return DataBase::GetOrCreateInstance()->m_map.at(name).get();
	}
	
	std::string DataBase::GetAllActNames()
	{
		DataBase* instance = DataBase::GetOrCreateInstance();
		std::string names;
		
		names.reserve(128); 
		std::string_view separator = "";
		for (const auto& key : std::views::keys(instance->m_map))
		{
			names += separator;
			names += key;
			separator = ", ";
		}

		return names;
	}
}
