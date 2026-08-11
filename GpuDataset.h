#pragma once
#include "Types.h"
#include <vector>

class GpuDataset 
{
public:
    GpuDataset(size_t totalPixels, size_t inputChannels, size_t targetChannels);
    ~GpuDataset();
    
    GpuDataset(const GpuDataset&) = delete;
    GpuDataset& operator=(const GpuDataset&) = delete;

    // Uploads the full dataset from CPU to GPU
    void UploadData(
        const std::vector<engineFloat>& inputAct, const std::vector<engineFloat>& inputGradX, const std::vector<engineFloat>& inputGradY,
        const std::vector<engineFloat>& targetAct, const std::vector<engineFloat>& targetGradX, const std::vector<engineFloat>& targetGradY
    );

    // Device Pointers for the entire dataset
    engineFloat* d_inputAct = nullptr;
    engineFloat* d_inputGradX = nullptr;
    engineFloat* d_inputGradY = nullptr;
    
    engineFloat* d_targetAct = nullptr;
    engineFloat* d_targetGradX = nullptr;
    engineFloat* d_targetGradY = nullptr;

private:
    size_t m_inputBytes;
    size_t m_targetBytes;
};