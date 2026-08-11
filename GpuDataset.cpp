#include "GpuDataset.h"
#include "cublas_utils.h"

GpuDataset::GpuDataset(size_t totalPixels, size_t inputChannels, size_t targetChannels)
{
    m_inputBytes = totalPixels * inputChannels * sizeof(engineFloat);
    m_targetBytes = totalPixels * targetChannels * sizeof(engineFloat);

    CUDA_CHECK(cudaMalloc((void**)&d_inputAct, m_inputBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_inputGradX, m_inputBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_inputGradY, m_inputBytes));

    CUDA_CHECK(cudaMalloc((void**)&d_targetAct, m_targetBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_targetGradX, m_targetBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_targetGradY, m_targetBytes));
}

GpuDataset::~GpuDataset()
{
    cudaFree(d_inputAct);
    cudaFree(d_inputGradX);
    cudaFree(d_inputGradY);
    cudaFree(d_targetAct);
    cudaFree(d_targetGradX);
    cudaFree(d_targetGradY);
}

void GpuDataset::UploadData(
    const std::vector<engineFloat>& inputAct, const std::vector<engineFloat>& inputGradX, const std::vector<engineFloat>& inputGradY,
    const std::vector<engineFloat>& targetAct, const std::vector<engineFloat>& targetGradX, const std::vector<engineFloat>& targetGradY)
{
    CUDA_CHECK(cudaMemcpy(d_inputAct, inputAct.data(), m_inputBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputGradX, inputGradX.data(), m_inputBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputGradY, inputGradY.data(), m_inputBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_targetAct, targetAct.data(), m_targetBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targetGradX, targetGradX.data(), m_targetBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targetGradY, targetGradY.data(), m_targetBytes, cudaMemcpyHostToDevice));
}
