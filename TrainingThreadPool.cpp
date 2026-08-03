#include "TrainingThreadPool.h"
#include "Network.h"
#include "Parameters.h"

TrainingThreadPool::TrainingThreadPool(size_t numThreads, Network& net) : m_network(net)
{
    m_workers.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) 
    {
        m_workers.emplace_back([this, i, numThreads]() 
        {
            // Allocate thread-local buffers ONCE per thread lifetime
            Network::SpatialDerivativeBuffer spatialBuffers;
            std::vector<Parameters> localDeltas = m_network.CreateEmptyDeltaBuffer();
            size_t localNumStored = 0;
            uint64_t localBatchID = 0; // Tracks which batch this thread completed last

            while (true) 
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                
                // Sleep until the global m_batchID increments or we stop
                m_cv_start.wait(lock, [this, localBatchID] { 
                    return (m_batchID > localBatchID) || m_stop; 
                });
                    
                if (m_stop) break;

                // Update local batch ID so this worker cannot re-run the same batch
                localBatchID = m_batchID;

                // Grab the job type and calculate chunk sizes
                PoolJobType currentJobType = m_jobType;
                size_t totalJobs = (currentJobType == PoolJobType::Training) ? m_batchSize : m_customTotalJobs;
                
                size_t jobsPerThread = totalJobs / numThreads;
                size_t startIdx = i * jobsPerThread;
                size_t endIdx = (i == numThreads - 1) ? totalJobs : startIdx + jobsPerThread;

                // Cache batch metadata locally while holding lock
                size_t startIndex = m_startIndex;
                const auto* inputs = m_inputs;
                const auto* targets = m_targets;
                auto customTask = m_customTask;

                lock.unlock(); // Unlock immediately for lock-free parallel execution

                if (currentJobType == PoolJobType::Training)
                {
                    // Reset thread-local accumulators
                    for(auto& p : localDeltas) p.Clear();
                    localNumStored = 0;
                    float localCost = 0.0f;

                    // Heavy math pass
                    for (size_t j = startIdx; j < endIdx; ++j) 
                    {
                        size_t dataIdx = (startIndex + j) % inputs->size(); 
                        const auto& input = (*inputs)[dataIdx];
                        const auto& target = (*targets)[dataIdx];

                        m_network.PropagateSpatialDerivativesThreadSafe(input.values, input.gradX, input.gradY, spatialBuffers);
                        m_network.BackPropagateGradientGuided(target, input.values, spatialBuffers.activations, spatialBuffers.preActivations, spatialBuffers, 0.0f, localDeltas, localNumStored);

                        float pixelCost = 0.0f;
                        for(size_t c = 0; c < target.values.size(); ++c){
                            float diff = spatialBuffers.activations.back()[c] - target.values[c];
                            pixelCost += diff * diff;
                        }
                        localCost += pixelCost / float(target.values.size());
                        
                        #ifndef _TRAINING
                        if ( std::_Is_nan(localCost) )
                        {
                            __debugbreak();
                        }
                        #endif
                    }

                    // Lock briefly to aggregate results
                    lock.lock();
                    m_network.AccumulateWorkerDeltas(localDeltas, localNumStored);
                    m_batchCost += localCost;
                }
                else if (currentJobType == PoolJobType::Custom)
                {
                    // Execute generic rendering loop safely on this thread
                    if (customTask) 
                    {
                        customTask(i, startIdx, endIdx);
                    }
                    lock.lock(); // Re-acquire lock to safely decrement active workers
                }

                m_activeWorkers--;
                    
                // Last worker to finish wakes up the main thread
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
                                    const std::vector<ImageUtils::SpatialData>& targets, 
                                    size_t startIndex, 
                                    size_t batchSize)
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_jobType = PoolJobType::Training; // Ensure threads know to do math
        m_inputs = &inputs;
        m_targets = &targets;
        m_startIndex = startIndex;
        m_batchSize = batchSize;
        m_batchCost = 0.0f;
        m_activeWorkers = m_workers.size();
        m_batchID++; // Increment generation counter to signal new work
    }
        
    m_cv_start.notify_all(); // Wake up all sleeping workers

    // Wait for all workers to finish the current generation
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv_done.wait(lock, [this] { return m_activeWorkers == 0; });

    return m_batchCost;
}

void TrainingThreadPool::RunParallelTask(size_t totalJobs, std::function<void(size_t threadIndex, size_t startIdx, size_t endIdx)> task)
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_jobType = PoolJobType::Custom; // Switch pool to render mode
        m_customTask = task;
        m_customTotalJobs = totalJobs;
        m_activeWorkers = m_workers.size();
        m_batchID++; 
    }
    
    m_cv_start.notify_all();
    
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv_done.wait(lock, [this] { return m_activeWorkers == 0; });
}