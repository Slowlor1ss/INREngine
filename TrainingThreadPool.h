#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>

#include "ImageUtils.h"

class Network;

class TrainingThreadPool 
{
public:
    TrainingThreadPool(size_t numThreads, Network& net);

    ~TrainingThreadPool();

    // Main thread calls this to fire off a batch across all CPU cores
    float RunBatch(const std::vector<ImageUtils::SpatialData>& inputs,
                   const std::vector<ImageUtils::SpatialData>& targets,
                   size_t startIndex,
                   size_t batchSize);

private:
    Network& m_network;
    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_cv_start;
    std::condition_variable m_cv_done;
    
    bool m_stop = false;
    bool m_startBatch = false;
    int m_activeWorkers = 0;
    float m_batchCost = 0.0f;

    const std::vector<ImageUtils::SpatialData>* m_inputs = nullptr;
    const std::vector<ImageUtils::SpatialData>* m_targets = nullptr;
    size_t m_startIndex = 0;
    size_t m_batchSize = 0;
};
