#pragma once

#include <sstream>

class Serializable
{
public:
	virtual void Serialize(std::ostream& out) = 0;
	virtual void Deserialize(std::istream& in) = 0;
};