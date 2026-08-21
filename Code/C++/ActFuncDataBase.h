#pragma once
#include <string>
#include <memory>
#include <map>
#include "ActivationFunctions.h"

namespace ActFunc
{
	class DataBase
	{
		static DataBase* m_instance;
		std::map<std::string, std::unique_ptr<Base>> m_map;

		static DataBase* GetOrCreateInstance();
	public:
		static Base* FindActFunc(const std::string& name);
		static std::string GetAllActNames();

		template <typename Imp>
		static Base* FindActFunc();
	};

	template<typename Imp>
	inline Base* DataBase::FindActFunc()
	{
		return DataBase::FindActFunc(Imp::k_name);
	}

}