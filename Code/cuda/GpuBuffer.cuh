#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include "cublas_utils.h"

// Quick litle RAII gpu buffer
template <typename T>
class GpuBuffer
{
private:
    T* d_ptr = nullptr; // Device pointer
    size_t m_elements = 0;

public:
    GpuBuffer() = default;
    // RAII :)
    ~GpuBuffer() { Free(); }

    // Note for those who come after, always ALWAYS implement your rule of 5 
    // I had such a painfull time debugging this bug and it would all ahve been solved if i ahs just followed the rule of 5 *sighs*
    // Disable copying (Prevents shallow copy double-frees)
    GpuBuffer( const GpuBuffer & ) = delete;
    GpuBuffer & operator=( const GpuBuffer & ) = delete;

    // Enable moving
    GpuBuffer( GpuBuffer && other ) noexcept
        : d_ptr( other.d_ptr ), m_elements( other.m_elements ) {
        other.d_ptr = nullptr;
        other.m_elements = 0;
    }

    GpuBuffer & operator=( GpuBuffer && other ) noexcept {
        if ( this != &other ) {
            Free(); // Clean up existing memory

            d_ptr = other.d_ptr;
            m_elements = other.m_elements;

            other.d_ptr = nullptr;
            other.m_elements = 0;
        }
        return *this;
    }

    // Allocates memory directly on the GPU VRAM
    void Allocate(size_t numElements)
    {
        Free(); // Ensure we don't leak memory if reused
        m_elements = numElements;
        if (numElements > 0)
        {
            CUDA_CHECK(cudaMalloc((void**)&d_ptr, numElements * sizeof(T)));
        }
    }
    
    void Free()
    {
        if (d_ptr)
        {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
            m_elements = 0;
        }
    }

    // Upload data from your CPU vector to GPU
    void Upload(const std::vector<T>& cpuData)
    {
        if (cpuData.size() != m_elements) Allocate(cpuData.size());
        CUDA_CHECK(cudaMemcpy(d_ptr, cpuData.data(), m_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Read data from GPU to CPU vector
    void Readback(std::vector<T>& cpuData) const
    {
        if (cpuData.size() != m_elements) cpuData.resize(m_elements);
        CUDA_CHECK( cudaMemcpy(cpuData.data(), d_ptr, m_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    T* Get() const { return d_ptr; }
    size_t Size() const { return m_elements; }
};