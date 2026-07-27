#pragma once
#include <string>
#include <memory>
#include <map>
#include "CostFunctions.h"

namespace CostFunc
{
	class DataBase
	{
		static DataBase* m_instance;
		std::map<std::string, std::unique_ptr<Base>> m_map;

		static DataBase* GetOrCreateInstance();
	public:
		static Base* FindCostFunc(const std::string& name);

		template <typename Imp>
		static Base* FindCostFunc();
	};

	template<typename Imp>
	inline Base* DataBase::FindCostFunc()
	{
		return DataBase::FindCostFunc(Imp::k_name);
	}
}
