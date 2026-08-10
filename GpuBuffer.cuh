#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// Quick litle RAII gpu buffer
template <typename T>
class GpuBuffer
{
private:
    T* d_ptr = nullptr; // Device pointer
    size_t m_elements = 0;

public:
    GpuBuffer() = default;

    // Allocates memory directly on the GPU VRAM
    void Allocate(size_t numElements)
    {
        Free(); // Ensure we don't leak memory if reused
        m_elements = numElements;
        if (numElements > 0)
        {
            cudaError_t err = cudaMalloc((void**)&d_ptr, numElements * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA Malloc failed!");
            }
        }
    }
    
    void Free()
    {
        if (d_ptr)
        {
            cudaFree(d_ptr);
            d_ptr = nullptr;
            m_elements = 0;
        }
    }

    // RAII :)
    ~GpuBuffer() { Free(); }

    // Upload data from your CPU vector to GPU
    void Upload(const std::vector<T>& cpuData)
    {
        if (cpuData.size() != m_elements) Allocate(cpuData.size());
        cudaMemcpy(d_ptr, cpuData.data(), m_elements * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Read data from GPU to CPU vector
    void Readback(std::vector<T>& cpuData) const
    {
        if (cpuData.size() != m_elements) cpuData.resize(m_elements);
        cudaMemcpy(cpuData.data(), d_ptr, m_elements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    T* Get() const { return d_ptr; }
    size_t Size() const { return m_elements; }
};