#include "TrainingThreadPool.h"
#include "Network.h"
#include "Parameters.h"

TrainingThreadPool::TrainingThreadPool(size_t numThreads, Network& net): m_network(net)
{
	m_workers.reserve(numThreads);
	for (size_t i = 0; i < numThreads; ++i) 
	{
		m_workers.emplace_back([this, i, numThreads]() 
		{
			// 1. Allocate thread-local memory EXACTLY ONCE
			Network::SpatialDerivativeBuffer spatialBuffers;
			std::vector<Parameters> localDeltas = m_network.CreateEmptyDeltaBuffer();
			size_t localNumStored = 0;

			while (true) 
			{
				// 2. Sleep until the main thread signals a new batch
				std::unique_lock<std::mutex> lock(m_mutex);
				m_cv_start.wait(lock, [this] { return m_startBatch || m_stop; });
                    
				if (m_stop) break;

				// 3. Calculate this thread's chunk of the batch
				size_t totalJobs = m_batchSize;
				size_t jobsPerThread = totalJobs / numThreads;
				size_t startIdx = i * jobsPerThread;
				size_t endIdx = (i == numThreads - 1) ? totalJobs : startIdx + jobsPerThread;

				lock.unlock(); // Unlock immediately so other threads can wake up

				// 4. Clear local deltas for the new batch
				for(auto& p : localDeltas) p.Clear();
				localNumStored = 0;
				float localCost = 0.0f;

				// 5. DO THE HEAVY MATH LOCK-FREE!
				for (size_t j = startIdx; j < endIdx; ++j) 
				{
					// Wrap around if the batch is larger than the dataset
					size_t dataIdx = (m_startIndex + j) % m_inputs->size(); 
					const auto& input = (*m_inputs)[dataIdx];
					const auto& target = (*m_targets)[dataIdx];

					m_network.PropagateSpatialDerivativesThreadSafe(input.values, input.gradX, input.gradY, spatialBuffers);
					m_network.BackPropagateGradientGuided(target, input.values, spatialBuffers.activations, spatialBuffers.preActivations, spatialBuffers, 0.0f, localDeltas, localNumStored);

					// Calculate standard RGB cost
					float pixelCost = 0.0f;
					for(size_t c = 0; c < target.values.size(); ++c){
						float diff = spatialBuffers.activations.back()[c] - target.values[c];
						pixelCost += diff * diff;
					}
#ifndef _TRAINING
				    if ( std::_Is_nan(cost) )
				    {
				        __debugbreak();
				    }
#endif
					localCost += pixelCost / float(target.values.size());
				}

				// 6. Lock briefly to sum the math into the global network
				lock.lock();
				m_network.AccumulateWorkerDeltas(localDeltas, localNumStored);
				m_batchCost += localCost;
				m_activeWorkers--;
                    
				// If I am the last thread to finish, wake up the main thread
				if (m_activeWorkers == 0) {
					m_cv_done.notify_one();
				}
			}
		});
	}
}

TrainingThreadPool::~TrainingThreadPool()
{
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_stop = true;
	}
	m_cv_start.notify_all();
	for (auto& worker : m_workers) {
		if (worker.joinable()) worker.join();
	}
}

float TrainingThreadPool::RunBatch(const std::vector<ImageUtils::SpatialData>& inputs,
	const std::vector<ImageUtils::SpatialData>& targets, size_t startIndex, size_t batchSize)
{
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_inputs = &inputs;
		m_targets = &targets;
		m_startIndex = startIndex;
		m_batchSize = batchSize;
		m_batchCost = 0.0f;
		m_activeWorkers = m_workers.size();
		m_startBatch = true;
	}
        
	m_cv_start.notify_all(); // WAKE UP ALL WORKERS!

	// Wait for all threads to finish
	std::unique_lock<std::mutex> lock(m_mutex);
	m_cv_done.wait(lock, [this] { return m_activeWorkers == 0; });
	m_startBatch = false;

	return m_batchCost;
}
