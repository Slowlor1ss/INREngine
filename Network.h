#pragma once
#include <vector>
#include <memory>
#include "Serializable.h"
#include "Parameters.h"
#include "Layer.h"

class Network : public Serializable
{
public:
	Network(const std::vector<size_t>& neuronsPerLayer, bool zeroInit = false);
	Network(Network&) = delete;
	Network(Network&&) = delete;
	Network& operator=(Network&) = delete;
	Network& operator=(Network&&) = delete;

	float CalculateCost(std::vector<float> inputActivation, std::vector<float> preferredOutput);
	std::vector<float> Propagate(std::vector<float> inputActivation);

	
	float BackPropagate(std::vector<float> inputActivation, std::vector<float> preferredOutput);

	void ConsumeDelta(float learningRate);

	// Inherited via Serializable
	virtual std::string Serialize() override;
	virtual void Deserialize(const std::string& inString) override;

private:
	void StoreDelta(const std::vector<Parameters>& other);
	InitialLayer& GetInitialLayer();
private:
	std::vector<std::unique_ptr<Layer>> m_layers;
	std::vector<Parameters> m_storedDelta;

	size_t m_numStored = 0;
};