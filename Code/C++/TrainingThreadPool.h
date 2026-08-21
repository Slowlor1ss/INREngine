#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <functional>

#include "ImageUtils.h"
#include "Types.h"

class Network;

enum class PoolJobType { Training, Custom };

class TrainingThreadPool 
{
public:
    TrainingThreadPool(size_t numThreads, Network& net);
    ~TrainingThreadPool();

    engineFloat RunBatch(const std::vector<ImageUtils::SpatialData>& inputs,
                   const std::vector<ImageUtils::SpatialData>& targets,
                   size_t startIndex,
                   size_t batchSize);

    // NEW: Generic Parallel Executor for Rendering
    void RunParallelTask(size_t totalJobs, std::function<void(size_t threadIndex, size_t startIdx, size_t endIdx)> task);

    // Helper to get thread count if needed for buffer allocation
    size_t GetThreadCount() const { return m_workers.size(); }

private:
    Network& m_network;
    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_cv_start;
    std::condition_variable m_cv_done;
    
    bool m_stop = false;
    uint64_t m_batchID = 0; 
    size_t m_activeWorkers = 0;
    engineFloat m_batchCost = 0.0f;

    // Training Data
    const std::vector<ImageUtils::SpatialData>* m_inputs = nullptr;
    const std::vector<ImageUtils::SpatialData>* m_targets = nullptr;
    size_t m_startIndex = 0;
    size_t m_batchSize = 0;

    // Custom Task Data
    PoolJobType m_jobType = PoolJobType::Training;
    std::function<void(size_t, size_t, size_t)> m_customTask;
    size_t m_customTotalJobs = 0;
};