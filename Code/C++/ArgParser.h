#pragma once

// ============================================================================
// Small argparse-style command line parser for C++.
// Based on Python's argparse: call AddArgument() / AddMultiValue() / AddAction() once per option, 
// then call Parse(argc, argv)
// ============================================================================

#include <algorithm>
#include <format>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class ArgParser
{
public:
	explicit ArgParser(std::string programDescription = "")
		: m_description(std::move(programDescription)) {}

	// Generic single-value option, e.g. "--lr 0.001"
	// ( Converter defaults to a plain stream extraction,
	// for simple values like float etc. pass [](const std::string& s){ return std::stof(s); } as converter )
	template <typename T>
	void AddArgument(std::string flag, std::string description, T& target,
	                 const std::function<T(const std::string&)>& converter = &ArgParser::DefaultConverter<T>,
	                 const std::vector<std::string>& extraHelpLines = {})
	{
		// Little "hack", This is just so we can capture flag before it gets moved
		auto handler = [&target, converter, flag](int& i, int argc, char** argv)
		{
			if (i + 1 >= argc) return;
			try
			{
				target = converter(argv[++i]);
			}
			catch (const std::exception& e)
			{
				std::cerr << "Error parsing value for " << flag << ": " << argv[i] << " Error: " << e.what() << "\n";
			}
		};
		
		m_options.push_back({
		  std::move(flag),
		  std::move(description),
		  std::move(handler),
		  extraHelpLines
	   });
	}

	// Multi-value option that keeps consuming args until the next "--"prefixed token
	// e.g. --layers 128, 64, 126 --next-arg ...
	template <typename T>
	void AddMultiValue(std::string flag, std::string description, std::vector<T>& target,
	                   const std::function<T(const std::string&)>& converter = &ArgParser::DefaultConverter<T>,
	                   const std::vector<std::string>& extraHelpLines = {})
	{
		// Little "hack", This is just so we can capture flag before it gets moved
		auto handler = [&target, converter, flag](int& i, int argc, char** argv)
		{
			target.clear();
			while (i + 1 < argc && argv[i + 1][0] != '-')
			{
				try
				{
					target.push_back(converter(argv[i + 1]));
				}
				catch (const std::exception& e)
				{
					std::cerr << "Error parsing value for " << flag << ": " << argv[i + 1] << " Error: " << e.what() << "\n";
				}
				++i;
			}
		};
		
		m_options.push_back({
			std::move(flag),
			std::move(description),
			std::move(handler),
			extraHelpLines
		});
	}

	// Custom handler for options whose parsing logic doesn't fit the generic
	// patterns of the other functions
	void AddAction(std::string flag, std::string description,
	               std::function<void(int& i, int argc, char** argv)> handler,
	               const std::vector<std::string>& extraHelpLines = {})
	{
		m_options.push_back({ std::move(flag), std::move(description), std::move(handler), extraHelpLines });
	}

	// Extra, seperate fucntion to add a note for the help print
	// printed under the most recently added option's description
	void AddHelpNote(std::string note)
	{
		if (!m_options.empty())
			m_options.back().extraHelpLines.push_back(std::move(note));
	}

	// Runs once after all args are parsed (if --help was not requested)
	// e.g. got summary printout
	void SetPostParseHook(std::function<void()> hook)
	{
		m_postParseHook = std::move(hook);
	}

	// For now just returns 1 to keep running, 0 if --help was shown (and we should exit)
	int Parse(int argc, char** argv) const
	{
		for (int i = 1; i < argc; ++i)
		{
			std::string arg = argv[i];
			if (arg == "--help" || arg == "--h" || arg == "-h" || arg == "help")
			{
				PrintHelp();
				return 0;
			}

			auto it = std::ranges::find_if(m_options, 
				[&arg](const Option& opt) { return opt.flag == arg; });

			if (it == m_options.end())
			{
				std::cout << "Unknown or incomplete argument: " << arg << "\n";
				continue;
			}

			it->handler(i, argc, argv);
		}

		if (m_postParseHook)
			m_postParseHook();

		return 1;
	}

	void PrintHelp() const
	{
		if (!m_description.empty())
			std::cout << m_description << "\n";

		std::cout << "Usage:\n";
		for (const auto& opt : m_options)
		{
			std::cout << std::format("  {:<14} | {}\n", opt.flag, opt.description);
			for (const auto& line : opt.extraHelpLines)
				std::cout << std::format("  {:<14} | {}\n", "", line);
		}
		std::cout << "==================================================================================================\n";
	}

private:
	template <typename T>
	static T DefaultConverter(const std::string& s)
	{
		std::istringstream iss(s);
		T value{};
		iss >> value;
		return value;
	}

	struct Option
	{
		std::string flag;
		std::string description;
		std::function<void(int&, int, char**)> handler;
		std::vector<std::string> extraHelpLines;
	};

	std::string m_description;
	std::vector<Option> m_options;
	std::function<void()> m_postParseHook;
};

// Specialized the default converters
template <>
inline float ArgParser::DefaultConverter<float>(const std::string& s) {
	return std::stof(s);
}

template <>
inline int ArgParser::DefaultConverter<int>(const std::string& s) {
	return std::stoi(s);
}

template <>
inline size_t ArgParser::DefaultConverter<size_t>(const std::string& s) {
	return std::stoull(s);
}

template <>
inline bool ArgParser::DefaultConverter<bool>(const std::string& s) {
	if (s == "1" || s == "true" || s == "True" || s == "TRUE") {
		return true;
	}
	if (s == "0" || s == "false" || s == "False" || s == "FALSE") {
		return false;
	}
	throw std::invalid_argument("Invalid boolean argument: " + s);
}