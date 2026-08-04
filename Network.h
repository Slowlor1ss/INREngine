#pragma once
#include <vector>
#include <memory>
#include "Serializable.h"
#include "Parameters.h"
#include "Layer.h"
#include "CostFuncDataBase.h"
#include "ImageUtils.h"

class Network : public Serializable
{
public:
	Network(
		const std::vector<size_t>& neuronsPerLayer,
		ActFunc::Base* hiddenActivation,
		ActFunc::Base* outputActivation,
		CostFunc::Base* costFunc = nullptr,
		bool zeroInit = false);
	Network(Network&) = delete;
	Network(Network&&) = delete;
	Network& operator=(Network&) = delete;
	Network& operator=(Network&&) = delete;

	void SetCostFunction(CostFunc::Base* costFunc) { m_costFunction = costFunc; }
	CostFunc::Base* GetCostFunction() const { return m_costFunction; }

	engineFloat CalculateCost(const std::vector<engineFloat>& inputActivation,const std::vector<engineFloat>& preferredOutput);
	std::vector<engineFloat> Propagate(const std::vector<engineFloat>& inputActivation);
	const std::vector<engineFloat>& PropagateThreadSafe(const std::vector<engineFloat>& inputActivation, std::vector<std::vector<engineFloat>>& threadBuffers) const;

	struct SpatialDerivativeBuffer
	{
		std::vector<std::vector<engineFloat>> preActivations;     // The raw 'z' sums before activation
	    std::vector<std::vector<engineFloat>> activations;		// The normal RGB values
	    std::vector<std::vector<engineFloat>> gradientX;			// How fast each output changes as X changes
	    std::vector<std::vector<engineFloat>> gradientY;			// How fast each output changes as Y changes
	};
	void PropagateSpatialDerivativesThreadSafe(
	    const std::vector<engineFloat>& inputActivation,
	    const std::vector<engineFloat>& inputGradX,
	    const std::vector<engineFloat>& inputGradY,
	    SpatialDerivativeBuffer& threadBuffers) const;
	
	engineFloat BackPropagate(const std::vector<engineFloat>& inputActivation,const std::vector<engineFloat>& preferredOutput);

	std::vector<Parameters> CreateEmptyDeltaBuffer() const { return m_storedDelta; }
	void AccumulateWorkerDeltas(const std::vector<Parameters>& workerDeltas, size_t workerNumStored);
	void BackPropagateGradientGuided(
	    const ImageUtils::SpatialData& target,
	    const std::vector<engineFloat>& input,
	    const std::vector<std::vector<engineFloat>>& forwardActivations,
	    const std::vector<std::vector<engineFloat>>& forwardSums,
	    const SpatialDerivativeBuffer& spatialBuffers,   // Network's predicted slopes
	    engineFloat learningRate,
	    std::vector<Parameters>& localDeltas,
	    size_t& localNumStored
	);
	
	void RunGradientGuidedEpoch(
		const std::vector<ImageUtils::SpatialData>& inputData, const std::vector<ImageUtils::SpatialData>& targetData, engineFloat
		learningRate);

	void ConsumeDelta(engineFloat learningRate);

	// Inherited via Serializable
	virtual void Serialize(std::ostream& out) override;
	virtual void Deserialize(std::istream& in) override;

private:
	void StoreDelta(const std::vector<Parameters>& other);
	InitialLayer& GetInitialLayer() const;
private:
	std::vector<std::unique_ptr<Layer>> m_layers;
	std::vector<Parameters> m_storedDelta;
	std::vector<std::vector<engineFloat>> m_costDeltas;
	CostFunc::Base* m_costFunction = nullptr;

	size_t m_numStored = 0;
};