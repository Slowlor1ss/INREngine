#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <iostream>

#include "cublas_utils.h"

// CudaContext singleton
class CudaManager
{
private:
    cublasHandle_t m_cublasHandle = nullptr;
    
    CudaManager()
    {
        CUBLAS_CHECK(cublasCreate(&m_cublasHandle));
        std::cout << "[GPU] cuBLAS Initialized Successfully.\n";
    }

    ~CudaManager()
    {
        if (m_cublasHandle)
        {
            CUBLAS_CHECK(cublasDestroy(m_cublasHandle));
            std::cout << "[GPU] cuBLAS Destroyed.\n";
        }
    }

public:
    // Delete copy/move semantics to enforce single instance
    CudaManager(const CudaManager&) = delete;
    CudaManager& operator=(const CudaManager&) = delete;

    // Global access point
    static CudaManager& GetInstance()
    {
        static CudaManager instance;
        return instance;
    }

    // Retrieve handle for math operations
    cublasHandle_t GetCublasHandle() const
    {
        return m_cublasHandle;
    }
};