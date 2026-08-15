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
        const std::vector<engineFloat>& targetAct, const std::vector<engineFloat>& targetGradX, const std::vector<engineFloat>& targetGradY,
        // Raw normalized [0,1] pixel coordinates, one entry per pixel -- this is what
        // feeds the grid encoder. Separate from inputAct/GradX/GradY (those are still
        // whatever the CPU coordMapper produces, e.g. PE features, if you're using one
        // alongside the grid encoder rather than in place of it).
        const std::vector<engineFloat>& pixelX, const std::vector<engineFloat>& pixelY
    );

    // Device Pointers for the entire dataset
    engineFloat* d_inputAct = nullptr;
    engineFloat* d_inputGradX = nullptr;
    engineFloat* d_inputGradY = nullptr;
    
    engineFloat* d_targetAct = nullptr;
    engineFloat* d_targetGradX = nullptr;
    engineFloat* d_targetGradY = nullptr;

    engineFloat* d_pixelX = nullptr;
    engineFloat* d_pixelY = nullptr;

private:
    size_t m_inputBytes;
    size_t m_targetBytes;
    size_t m_pixelBytes;
};