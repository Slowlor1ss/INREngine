#pragma once
#include <vector>
#include <memory>
#include "Serializable.h"
#include "Parameters.h"
#include "Layer.h"
#include "CostFuncDataBase.h"

class Network : public Serializable
{
public:

	struct LayerInfo
	{
		size_t numNeurons;
		ActFunc::Base* actFunc;
	};

	Network(const std::vector<LayerInfo>& layerInfos, CostFunc::Base* costFunc = nullptr, bool zeroInit = false);
	Network(Network&) = delete;
	Network(Network&&) = delete;
	Network& operator=(Network&) = delete;
	Network& operator=(Network&&) = delete;

	void SetCostFunction(CostFunc::Base* costFunc) { m_costFunction = costFunc; }
	CostFunc::Base* GetCostFunction() const { return m_costFunction; }

	float CalculateCost(const std::vector<float>& inputActivation,const std::vector<float>& preferredOutput);
	std::vector<float> Propagate(const std::vector<float>& inputActivation);
	const std::vector<float>& PropagateThreadSafe(const std::vector<float>& inputActivation, std::vector<std::vector<float>>& threadBuffers) const;

	
	float BackPropagate(const std::vector<float>& inputActivation,const std::vector<float>& preferredOutput);

	void ConsumeDelta(float learningRate);

	// Inherited via Serializable
	virtual void Serialize(std::ostream& out) override;
	virtual void Deserialize(std::istream& in) override;

private:
	void StoreDelta(const std::vector<Parameters>& other);
	InitialLayer& GetInitialLayer();
private:
	std::vector<std::unique_ptr<Layer>> m_layers;
	std::vector<Parameters> m_storedDelta;
	std::vector<std::vector<float>> m_costDeltas;
	CostFunc::Base* m_costFunction = nullptr;

	size_t m_numStored = 0;
};