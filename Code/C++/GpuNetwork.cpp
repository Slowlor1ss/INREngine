#include "GpuNetwork.h"
#include "cublas_utils.h"
#include <iostream>

#include "AdamOptimizer.cuh"
#include "BackwardPass.cuh"
#include "ForwardPass.cuh"
#include "GridEncoding.cuh"
#include <random>

// Main wrapper: Executes one full batch entirely on the GPU
void GpuNetwork::TrainBatchGPU(
    const engineFloat* d_batchInputAct,
    const engineFloat* d_batchInputGradX,
    const engineFloat* d_batchInputGradY,
    const engineFloat* d_batchTargetAct,
    const engineFloat* d_batchTargetGradX,
    const engineFloat* d_batchTargetGradY,
    const engineFloat* d_batchPixelXSrc,
    const engineFloat* d_batchPixelYSrc,
    engineFloat spatialLossWeight,
    engineFloat learningRate, 
    bool enableCoordJitter, 
    engineFloat jitterAmpX, 
    engineFloat jitterAmpY)
{
#ifndef _TRAINING
    {
        static int batchCounter = 0;
        cudaError_t ghostErr = cudaGetLastError();
        if ( ghostErr != cudaSuccess ) {
            printf( "\n[GHOST CAUGHT] Error arrived before Batch %d started: %s\n", batchCounter, cudaGetErrorString( ghostErr ) );
            __debugbreak();
        }
        batchCounter++;
    }
#endif
    
    // Update the dynamic LR buffer on the stream (not captured by the graph if done outside)
    //CUDA_CHECK(cudaMemcpyAsync(d_learningRate, &learningRate, sizeof(engineFloat), cudaMemcpyHostToDevice, m_stream));
    
    // Sources:
    // https://leimao.github.io/blog/Page-Locked-Host-Memory-Page-Table/#:~:text=This%20allows%20for%20faster%20data%20transfer%20between,page%2Dlocked%20memory%20to%20physical%20addresses%20in%20RAM.
    // https://www.abhik.ai/concepts/gpu-computing/cuda-streams
    // Explenation as to why we do this, if we were to do a async cpy of the learning rate passed in the function it can (and will) go out
    // of scope so we'd end up setting d_learnignRate to garbage or 0, instead we have now created a raw pointer to set the d_learningrate
    // we use a pointer allocated with cudaMallocHost (page-locked memory) instead of a (non pinned)-member variable as in theory this 
    // should prevent the copy from turning in to a blocking copy, as pageable memory could get moved around 
    // which due to some more complicated stuff can turn our async copy in to a blocking one; see sources above ^
    *h_learningRate = learningRate;
    CUDA_CHECK(cudaMemcpyAsync(d_learningRate, h_learningRate, sizeof(engineFloat), cudaMemcpyHostToDevice, m_stream));
    
    // Stage the Input Data (we use the layer 0 variables as a sort of fixed memory location 
    // that we constantly override and its purpose is to be able to pass in the parameters trough a fixed memory location, which we need due to how cuda graphs work)
    GpuLayer& inputLayer = m_layers[0];
    size_t inBytes = m_batchSize * inputLayer.numNeurons * sizeof(engineFloat);
    if (!m_useGridEncoding) {
        // Static coordMapper-encoded input path
        CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_activations, d_batchInputAct, inBytes, cudaMemcpyDeviceToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActFreq, d_batchInputAct, inBytes, cudaMemcpyDeviceToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_gradX, d_batchInputGradX, inBytes, cudaMemcpyDeviceToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_gradY, d_batchInputGradY, inBytes, cudaMemcpyDeviceToDevice, m_stream));
        if (inputLayer.hasDualWeights) {
            CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActScale, d_batchInputAct, inBytes, cudaMemcpyDeviceToDevice, m_stream));
        }
    }
    // else: inputLayer.d_activations/d_gradX/d_gradY get written by
    // RunGridEncodeForwardGPU at the top of ForwardPass() instead, every batch,
    // inside the captured graph.

    // Stage the Target Data (We added soem new fixed Buffers as we dont have a layer 0 equevalent here)
    size_t outBytes = m_batchSize * m_layers.back().numNeurons * sizeof(engineFloat);
    CUDA_CHECK(cudaMemcpyAsync(d_fixedTargetAct, d_batchTargetAct, outBytes, cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fixedTargetGradX, d_batchTargetGradX, outBytes, cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fixedTargetGradY, d_batchTargetGradY, outBytes, cudaMemcpyDeviceToDevice, m_stream));

    // Fixed mailbox like explained above
    if (m_useGridEncoding) {
        size_t pixelBytes = m_batchSize * sizeof(engineFloat);
        if (enableCoordJitter) {
            RunJitterPixelCoordsGPU(
                d_batchPixelXSrc, d_batchPixelYSrc,
                m_gridEncoder.d_batchPixelX, m_gridEncoder.d_batchPixelY,
                m_batchSize, jitterAmpX, jitterAmpY,
                m_jitterSeedCounter++, m_stream);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(m_gridEncoder.d_batchPixelX, d_batchPixelXSrc, pixelBytes, cudaMemcpyDeviceToDevice, m_stream));
            CUDA_CHECK(cudaMemcpyAsync(m_gridEncoder.d_batchPixelY, d_batchPixelYSrc, pixelBytes, cudaMemcpyDeviceToDevice, m_stream));
        }
    }
    
    if (!m_graphCaptured)
    {
        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal));
        // Cuda graphs has unfortinutly broken the profile markers keep it here for now but dont think its fixable due to how cuda graphs work :(
        PROFILE_PUSH_COLOR("ForwardPass", 0xFFFFB3BA); 
        // Pushes the batch of pixels through the network to calculate colors and spatial slopes.
        ForwardPass();
        PROFILE_POP();
        
        PROFILE_PUSH_COLOR("BackwardPass", 0xFFBAFFC9); 
        // Calculates the error against the target image and uses cublas to accumulate 
        // the gradients into the d_delta buffers.
        BackwardPass(spatialLossWeight);
        PROFILE_POP();

        PROFILE_PUSH_COLOR("ApplyGradientsGPU", 0xFFBAE1FF); 
        // Apply gradients (ADAM OPTIMIZER)
        // Updates all weights, momentum, and velocity in VRAM (auto-clears the deltas to 0.0f)
        ApplyGradientsGPU(learningRate);
        PROFILE_POP();
        
        CUDA_CHECK(cudaStreamEndCapture(m_stream, &m_graph));
        CUDA_CHECK(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));

        m_graphCaptured = true;
    }
    
    // Launch captured graph
    CUDA_CHECK(cudaGraphLaunch(m_graphExec, m_stream));
}

// Initialize cublas and mirror the CPU network structure to VRAM
GpuNetwork::GpuNetwork(const Network& cpuNetwork, int batchSize, bool useGridEncoding)
    : m_batchSize(batchSize), m_useGridEncoding(useGridEncoding)
{
    // Initialize the cublas Hardware Context
    CUBLAS_CHECK(cublasCreate(&m_cublasHandle));
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUBLAS_CHECK(cublasSetStream(m_cublasHandle, m_stream));
    
    m_costType = cpuNetwork.GetCostFunction()->GetGpuType();

    // Mirror every layer to the GPU
    const auto& cpuLayers = cpuNetwork.GetLayers(); 

    // Use an index-based loop so we can peek at layer [i - 1]
    for (size_t i = 0; i < cpuLayers.size(); ++i)
    {
        const auto* cpuLayer = cpuLayers[i].get();
        GpuLayer gpuLayer;

        gpuLayer.numNeurons = static_cast<int>(cpuLayer->GetNumNeurons());
        
        // TODO: this explicit split is probably not needed num neurons should be 0 when using getnumneurons but check to be sure
        if (i == 0) {
            gpuLayer.prevNeurons = 0; // Layer 0 has no previous layer
        } else {
            gpuLayer.prevNeurons = static_cast<int>(cpuLayers[i - 1]->GetNumNeurons());
        }
        
        if (cpuLayer->GetActivationFunction()) {
            gpuLayer.actType = cpuLayer->GetActivationFunction()->GetGpuType();
        } else {
            gpuLayer.actType = GpuActType::None;
        }
        
        gpuLayer.learningRateMultiplier = cpuLayer->GetLearningrateMultiplier();

        const auto& params = cpuLayer->GetParams();
        gpuLayer.hasDualWeights = !params.weights_scale.empty();

        // Allocate the VRAM for this layer
        AllocateLayerMemory(gpuLayer, m_batchSize);
        
        const size_t bSize = gpuLayer.numNeurons * sizeof(engineFloat);
        
        // Verify Biases
#ifndef _TRAINING
        if (params.biases.size() != static_cast<size_t>(gpuLayer.numNeurons)) {
            std::cout << "[FATAL GPU ERROR] Bias dimension mismatch!\n"
                      << "Layer Expected: " << gpuLayer.numNeurons << " biases, but CPU has: " << params.biases.size() << "\n";
            __debugbreak(); 
        }
#endif
        CUDA_CHECK(cudaMemcpy(gpuLayer.d_biases, params.biases.data(), bSize, cudaMemcpyHostToDevice));
        
        if (gpuLayer.hasDualWeights) {
#ifndef _TRAINING
            if (params.biases_scale.size() != static_cast<size_t>(gpuLayer.numNeurons)) {
                std::cout << "[FATAL GPU ERROR] Bias Scale dimension mismatch!\n";
                __debugbreak();
            }
#endif
            CUDA_CHECK(cudaMemcpy(gpuLayer.d_biasesScale, params.biases_scale.data(), bSize, cudaMemcpyHostToDevice));
        }

        // Verify Weights
        if (gpuLayer.prevNeurons > 0) 
        {
            const size_t expectedWeights = static_cast<size_t>(gpuLayer.numNeurons) * gpuLayer.prevNeurons;
            const size_t wSize = expectedWeights * sizeof(engineFloat);
#ifndef _TRAINING
            if (params.weights.size() != expectedWeights) {
                std::cout << "[FATAL GPU ERROR] Weight dimension mismatch!\n"
                          << "Layer Expected: " << expectedWeights << " weights, but CPU has: " << params.weights.size() << "\n";
                __debugbreak();
            }
#endif
            CUDA_CHECK(cudaMemcpy(gpuLayer.d_weights, params.weights.data(), wSize, cudaMemcpyHostToDevice));
            
            if (gpuLayer.hasDualWeights) {
#ifndef _TRAINING
                if (params.weights_scale.size() != expectedWeights) {
                    std::cout << "[FATAL GPU ERROR] Weight Scale dimension mismatch!\n";
                    __debugbreak();
                }
#endif
                CUDA_CHECK(cudaMemcpy(gpuLayer.d_weightsScale, params.weights_scale.data(), wSize, cudaMemcpyHostToDevice));
            }
        }

        m_layers.push_back(gpuLayer);
    }
    
    // Not per layer allocations:
    CUDA_CHECK(cudaMalloc(&d_learningRate, sizeof(engineFloat)));
    CUDA_CHECK(cudaMallocHost(&h_learningRate, sizeof(engineFloat)));
    
    // Target buffers matching the size of your final output layer
    const size_t targetBytes = m_batchSize * m_layers.back().numNeurons * sizeof(engineFloat);
    CUDA_CHECK(cudaMalloc(&d_fixedTargetAct, targetBytes));
    CUDA_CHECK(cudaMalloc(&d_fixedTargetGradX, targetBytes));
    CUDA_CHECK(cudaMalloc(&d_fixedTargetGradY, targetBytes));
    
    CUDA_CHECK(cudaMalloc(&d_totalCost, sizeof(engineFloat)));

    if (m_useGridEncoding) {
        AllocateGridEncoderMemory(m_batchSize);

#ifndef _TRAINING
        if (m_gridEncoder.totalOutputChannels != m_layers[0].numNeurons) {
            std::cout << "[FATAL GPU ERROR] Grid encoder output channel mismatch!\n"
                      << "Grid encoder produces: " << m_gridEncoder.totalOutputChannels
                      << " channels, but CPU Layer 0 (InitialLayer) has: " << m_layers[0].numNeurons << " neurons.\n"
                      << "Fix: size the CPU InitialLayer to GridEncoding::Config::NumLevels * FeaturesPerLevel.\n";
            __debugbreak();
        }
#endif
    }
}

// Clean up all VRAM to prevent memory leaks
GpuNetwork::~GpuNetwork()
{
    for (auto& layer : m_layers) {
        FreeLayerMemory(layer);
    }
    if (m_useGridEncoding) FreeGridEncoderMemory();
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
    cudaStreamDestroy(m_stream);
    cublasDestroy(m_cublasHandle);
}

// Chckpointing Download trained weights from VRAM back to CPU RAM
void GpuNetwork::DownloadParametersToCPU(Network& cpuNetwork)
{
    PROFILE_SCOPE("DownloadParametersToCPU");
    const auto& cpuLayers = cpuNetwork.GetLayers(); 

    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& gpuLayer = m_layers[i];
        auto* cpuLayer = cpuLayers[i].get();
        Parameters& cpuParams = cpuLayer->GetParams(); 

        size_t wSize = gpuLayer.numNeurons * gpuLayer.prevNeurons * sizeof(engineFloat);
        size_t bSize = gpuLayer.numNeurons * sizeof(engineFloat);

        CUDA_CHECK(cudaMemcpy((void**)cpuParams.weights.data(), gpuLayer.d_weights, wSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy((void**)cpuParams.biases.data(), gpuLayer.d_biases, bSize, cudaMemcpyDeviceToHost));

        if (gpuLayer.hasDualWeights) {
            CUDA_CHECK(cudaMemcpy((void**)cpuParams.weights_scale.data(), gpuLayer.d_weightsScale, wSize, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy((void**)cpuParams.biases_scale.data(), gpuLayer.d_biasesScale, bSize, cudaMemcpyDeviceToHost));
        }
    }
}

engineFloat GpuNetwork::GetLastBatchCost(const engineFloat* d_batchTargetAct)
{
    // Read directly from the final layer's activation buffer!
    const engineFloat* d_finalActivations = m_layers.back().d_activations;
    int numOutputNeurons = m_layers.back().numNeurons;

    return CalculateBatchCostGPU(
        m_batchSize, numOutputNeurons, 
        d_finalActivations, d_batchTargetAct, 
        d_totalCost, m_costType
    );
}

// Forward pass returning predicted colors to the CPU for rendering
void GpuNetwork::PredictGPU(const engineFloat* d_predictPixelX, const engineFloat* d_predictPixelY, const engineFloat* d_standardInputs, engineFloat* d_predictOutputs, int totalPixels)
{
    GpuLayer& inputLayer = m_layers[0];
    GpuLayer& outputLayer = m_layers.back();

    for (int offset = 0; offset < totalPixels; offset += m_batchSize)
    {
        int currentChunk = std::min((int)m_batchSize, totalPixels - offset);
        
        if (m_useGridEncoding) {
            size_t pixelBytes = currentChunk * sizeof(engineFloat);
            CUDA_CHECK(cudaMemcpyAsync(m_gridEncoder.d_batchPixelX, d_predictPixelX + offset, pixelBytes, cudaMemcpyDeviceToDevice, m_stream));
            CUDA_CHECK(cudaMemcpyAsync(m_gridEncoder.d_batchPixelY, d_predictPixelY + offset, pixelBytes, cudaMemcpyDeviceToDevice, m_stream));
        } else {
            int inPointerOffset = offset * inputLayer.numNeurons;
            size_t inBytes = currentChunk * inputLayer.numNeurons * sizeof(engineFloat);
            CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_activations, d_standardInputs + inPointerOffset, inBytes, cudaMemcpyDeviceToDevice, m_stream));
            CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActFreq, d_standardInputs + inPointerOffset, inBytes, cudaMemcpyDeviceToDevice, m_stream));
        }

        ForwardPass();

        int outPointerOffset = offset * outputLayer.numNeurons;
        size_t outBytes = currentChunk * outputLayer.numNeurons * sizeof(engineFloat);
        CUDA_CHECK(cudaMemcpyAsync(d_predictOutputs + outPointerOffset, outputLayer.d_activations, outBytes, cudaMemcpyDeviceToDevice, m_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

// Request raw memory from the GPU
void GpuNetwork::AllocateLayerMemory(GpuLayer& layer, int batchSize)
{
    PROFILE_SCOPE_FMT("AllocateLayerMemory - layer size %d", layer.numNeurons);
    const size_t batchNeuronBytes = batchSize * layer.numNeurons * sizeof(engineFloat);
    const size_t biasBytes = layer.numNeurons * sizeof(engineFloat);
    const size_t weightBytes = layer.numNeurons * layer.prevNeurons * sizeof(engineFloat);
    
    // Persistent Parameters & Deltas (Biases)
    CUDA_CHECK(cudaMalloc(&layer.d_biases, biasBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_deltaBiases, biasBytes));
    CUDA_CHECK(cudaMemset(layer.d_deltaBiases, 0, biasBytes)); // Must start at 0 for accumulation!

    if (layer.hasDualWeights) {
        CUDA_CHECK(cudaMalloc(&layer.d_biasesScale, biasBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaBiasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_deltaBiasesScale, 0, biasBytes));
    }
    
    // Allocate Adam States (Biases)
    CUDA_CHECK(cudaMalloc(&layer.d_m_biases, biasBytes));
    CUDA_CHECK(cudaMemset(layer.d_m_biases, 0, biasBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_v_biases, biasBytes));
    CUDA_CHECK(cudaMemset(layer.d_v_biases, 0, biasBytes));

    if (layer.hasDualWeights) {
        CUDA_CHECK(cudaMalloc(&layer.d_biasesScale, biasBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaBiasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_deltaBiasesScale, 0, biasBytes));
        
        // Adam States (Biases Scale)
        CUDA_CHECK(cudaMalloc(&layer.d_m_biasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_m_biasesScale, 0, biasBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_v_biasesScale, biasBytes));
        CUDA_CHECK(cudaMemset(layer.d_v_biasesScale, 0, biasBytes));
    }
    
    if (layer.prevNeurons > 0) {
        CUDA_CHECK(cudaMalloc(&layer.d_weights, weightBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaWeights, weightBytes));
        CUDA_CHECK(cudaMemset(layer.d_deltaWeights, 0, weightBytes)); 

        // Adam States (Weights Freq)
        CUDA_CHECK(cudaMalloc(&layer.d_m_weights, weightBytes));
        CUDA_CHECK(cudaMemset(layer.d_m_weights, 0, weightBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_v_weights, weightBytes));
        CUDA_CHECK(cudaMemset(layer.d_v_weights, 0, weightBytes));

        if (layer.hasDualWeights) {
            CUDA_CHECK(cudaMalloc(&layer.d_weightsScale, weightBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_deltaWeightsScale, weightBytes));
            CUDA_CHECK(cudaMemset(layer.d_deltaWeightsScale, 0, weightBytes));
            
            // Adam States (Weights Scale)
            CUDA_CHECK(cudaMalloc(&layer.d_m_weightsScale, weightBytes));
            CUDA_CHECK(cudaMemset(layer.d_m_weightsScale, 0, weightBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_v_weightsScale, weightBytes));
            CUDA_CHECK(cudaMemset(layer.d_v_weightsScale, 0, weightBytes));
        }
    }

    // Forward Pass Buffers
    CUDA_CHECK(cudaMalloc(&layer.d_preActFreq, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_activations, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_gradX, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_gradY, batchNeuronBytes));

    if (layer.hasDualWeights) {
	    CUDA_CHECK(cudaMalloc(&layer.d_preActScale, batchNeuronBytes));
	    CUDA_CHECK(cudaMalloc(&layer.d_rawSlopeXScale, batchNeuronBytes));
	    CUDA_CHECK(cudaMalloc(&layer.d_rawSlopeYScale, batchNeuronBytes));
	}

    // Backward Pass Buffers (Factored Terms)
    if (layer.prevNeurons > 0) {
        CUDA_CHECK(cudaMalloc(&layer.d_deltaAFreq, batchNeuronBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaXFreq, batchNeuronBytes));
        CUDA_CHECK(cudaMalloc(&layer.d_deltaYFreq, batchNeuronBytes));

        if (layer.hasDualWeights) {
            CUDA_CHECK(cudaMalloc(&layer.d_deltaAScale, batchNeuronBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_deltaXScale, batchNeuronBytes));
            CUDA_CHECK(cudaMalloc(&layer.d_deltaYScale, batchNeuronBytes));
        }
    }

    // Backward Pass Buffers (Incoming Errors)
    CUDA_CHECK(cudaMalloc(&layer.d_colorError, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_errorGradX, batchNeuronBytes));
    CUDA_CHECK(cudaMalloc(&layer.d_errorGradY, batchNeuronBytes));
}

// Allocate the grid encoder's per-level buffers and initialize its parameters
void GpuNetwork::AllocateGridEncoderMemory(int batchSize)
{
    PROFILE_SCOPE("AllocateGridEncoderMemory");
    using namespace GridEncoding;

    m_gridEncoder.numLevels = Config::NumLevels;
    m_gridEncoder.featuresPerLevel = Config::FeaturesPerLevel;
    m_gridEncoder.totalOutputChannels = Config::NumLevels * Config::FeaturesPerLevel;

    // Work out each level's resolution + flat offset into the concatenated
    // parameter buffer, once, on the host.
    std::vector<int> levelResolutions(Config::NumLevels);
    std::vector<int> levelParamOffsets(Config::NumLevels);
    int runningOffset = 0;
    for (int level = 0; level < Config::NumLevels; ++level) {
        levelResolutions[level] = LevelResolution(level);
        levelParamOffsets[level] = runningOffset;
        runningOffset += LevelNumParams(level);
    }
    m_gridEncoder.totalParams = runningOffset;

    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_levelResolutions, Config::NumLevels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_levelParamOffsets, Config::NumLevels * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(m_gridEncoder.d_levelResolutions, levelResolutions.data(), Config::NumLevels * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m_gridEncoder.d_levelParamOffsets, levelParamOffsets.data(), Config::NumLevels * sizeof(int), cudaMemcpyHostToDevice));

    const size_t paramBytes = m_gridEncoder.totalParams * sizeof(engineFloat);
    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_params, paramBytes));
    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_grad, paramBytes));
    CUDA_CHECK(cudaMemset(m_gridEncoder.d_grad, 0, paramBytes)); // must start at 0, same reasoning as d_deltaWeights
    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_m, paramBytes));
    CUDA_CHECK(cudaMemset(m_gridEncoder.d_m, 0, paramBytes));
    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_v, paramBytes));
    CUDA_CHECK(cudaMemset(m_gridEncoder.d_v, 0, paramBytes));

    // Small random init (Instant-NGP inits near 0, these features get refined
    // fast, and starting large risks the first few batches producing noisy,
    // high-magnitude activations into the MLP before anything has trained)
    std::vector<engineFloat> hostInit(m_gridEncoder.totalParams);
    std::mt19937 gen(1337);
    std::uniform_real_distribution<engineFloat> dist(-1e-4f, 1e-4f);
    for (auto& v : hostInit) v = dist(gen);
    CUDA_CHECK(cudaMemcpy(m_gridEncoder.d_params, hostInit.data(), paramBytes, cudaMemcpyHostToDevice));

    const size_t pixelBytes = batchSize * sizeof(engineFloat);
    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_batchPixelX, pixelBytes));
    CUDA_CHECK(cudaMalloc(&m_gridEncoder.d_batchPixelY, pixelBytes));
}

void GpuNetwork::FreeGridEncoderMemory()
{
    auto SafeFree = [](engineFloat*& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } };
    auto SafeFreeInt = [](int*& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } };

    SafeFreeInt(m_gridEncoder.d_levelResolutions);
    SafeFreeInt(m_gridEncoder.d_levelParamOffsets);
    SafeFree(m_gridEncoder.d_params);
    SafeFree(m_gridEncoder.d_grad);
    SafeFree(m_gridEncoder.d_m);
    SafeFree(m_gridEncoder.d_v);
    SafeFree(m_gridEncoder.d_batchPixelX);
    SafeFree(m_gridEncoder.d_batchPixelY);
}

// Return memory to the GPU
void GpuNetwork::FreeLayerMemory(GpuLayer& layer)
{
    PROFILE_SCOPE_FMT("FreeLayerMemory - layer size %d", layer.numNeurons);
    auto SafeFree = [](engineFloat*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    // These are not per layer but as we check the pointer this should be fine
    SafeFree(d_learningRate);
    // Not using SafeFree as this is host memory and needs cudaFreeHost
    if (h_learningRate) {
        cudaFreeHost(h_learningRate);
        h_learningRate = nullptr;
    }
    
    SafeFree(d_fixedTargetAct);
    SafeFree(d_fixedTargetGradX);
    SafeFree(d_fixedTargetGradY);
    
    SafeFree(d_totalCost);
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    // Parameters & Deltas
    SafeFree(layer.d_weights);
    SafeFree(layer.d_weightsScale);
    SafeFree(layer.d_biases);
    SafeFree(layer.d_biasesScale);

    SafeFree(layer.d_deltaWeights);
    SafeFree(layer.d_deltaWeightsScale);
    SafeFree(layer.d_deltaBiases);
    SafeFree(layer.d_deltaBiasesScale);

    // Adam States
    SafeFree(layer.d_m_weights);
    SafeFree(layer.d_v_weights);
    SafeFree(layer.d_m_biases);
    SafeFree(layer.d_v_biases);

    SafeFree(layer.d_m_weightsScale);
    SafeFree(layer.d_v_weightsScale);
    SafeFree(layer.d_m_biasesScale);
    SafeFree(layer.d_v_biasesScale);

    // Forward Buffers
    SafeFree(layer.d_preActFreq);
    SafeFree(layer.d_preActScale);
    SafeFree(layer.d_activations);
    SafeFree(layer.d_gradX);
    SafeFree(layer.d_gradY);
    SafeFree(layer.d_rawSlopeXScale);
	SafeFree(layer.d_rawSlopeYScale);

    // Backward Buffers
    SafeFree(layer.d_deltaAFreq);
    SafeFree(layer.d_deltaXFreq);
    SafeFree(layer.d_deltaYFreq);
    SafeFree(layer.d_deltaAScale);
    SafeFree(layer.d_deltaXScale);
    SafeFree(layer.d_deltaYScale);

    // Error Buffers
    SafeFree(layer.d_colorError);
    SafeFree(layer.d_errorGradX);
    SafeFree(layer.d_errorGradY);
}

void GpuNetwork::ForwardPass()
{
    if (m_useGridEncoding) {
        GpuLayer& inputLayer = m_layers[0];
        PROFILE_PUSH_COLOR("GridEncodeForward", 0xFFFFE1BA);
        RunGridEncodeForwardGPU(
            m_stream, m_batchSize, m_gridEncoder.numLevels, m_gridEncoder.featuresPerLevel,
            m_gridEncoder.d_levelResolutions, m_gridEncoder.d_levelParamOffsets,
            m_gridEncoder.d_params,
            m_gridEncoder.d_batchPixelX, m_gridEncoder.d_batchPixelY,
            inputLayer.d_activations, inputLayer.d_gradX, inputLayer.d_gradY
        );
        // Layer 0 has no activation function (identity), so preActFreq == activations,
        // same convention the old static-input path used.
        size_t inBytes = m_batchSize * inputLayer.numNeurons * sizeof(engineFloat);
        CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActFreq, inputLayer.d_activations, inBytes, cudaMemcpyDeviceToDevice, m_stream));
        PROFILE_POP();
    }

    // Layer 0 (Input Passthrough)
    //GpuLayer& inputLayer = m_layers[0];
    //size_t batchNeuronBytes = m_batchSize * inputLayer.numNeurons * sizeof(engineFloat);

    // Copy the raw batch inputs directly into Layer 0's forward buffers
    //CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_activations, d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    //CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActFreq,  d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    //CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_gradX,       d_batchInputGradX, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    //CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_gradY,       d_batchInputGradY, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));

    //if (inputLayer.hasDualWeights) {
    //    CUDA_CHECK(cudaMemcpyAsync(inputLayer.d_preActScale, d_batchInputAct, batchNeuronBytes, cudaMemcpyDeviceToDevice, m_stream));
    //}
 
    // Forward propagation loop
    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& curr = m_layers[i];
        GpuLayer& prev = m_layers[i - 1];
        
        PROFILE_PUSH_FMT("Forward propagation loop - layer size %d", curr.numNeurons);
        // Launch the Forward Kernel for this specific layer
        RunForwardLayerGPU(
            m_cublasHandle, m_stream,
            m_batchSize, curr.numNeurons, prev.numNeurons,
            curr.d_weights, curr.d_weightsScale, curr.d_biases, curr.d_biasesScale,
            prev.d_activations, prev.d_gradX, prev.d_gradY,
            curr.d_preActFreq, curr.d_preActScale,
            curr.d_activations, curr.d_gradX, curr.d_gradY,
            curr.d_rawSlopeXScale, curr.d_rawSlopeYScale,
            curr.actType, curr.hasDualWeights
        );
        PROFILE_POP();
    }
}

void GpuNetwork::BackwardPass(
    engineFloat spatialLossWeight)
{
    // Calculate output error
    GpuLayer& outputLayer = m_layers.back();
    
    CalculateOutputErrorGPU(
        m_stream,
        m_batchSize, outputLayer.numNeurons,
        outputLayer.d_activations, outputLayer.d_gradX, outputLayer.d_gradY,
        d_fixedTargetAct, d_fixedTargetGradX, d_fixedTargetGradY,
        outputLayer.d_colorError, outputLayer.d_errorGradX, outputLayer.d_errorGradY,
        spatialLossWeight, m_costType
    );
    
    // Update All Layers (Looping from Output down to Layer 0)
    for (int i = (int)m_layers.size() - 1; i > 0; --i)
    {
        GpuLayer& curr = m_layers[i];
        GpuLayer& prev = m_layers[i - 1];

        PROFILE_PUSH_FMT("Backward propagation loop - layer size %d", curr.numNeurons);
        // We must wipe the previous layer's error buffers to zero before cublas accumulates into them
        size_t batchNeuronBytes = m_batchSize * prev.numNeurons * sizeof(engineFloat);
        CUDA_CHECK(cudaMemsetAsync(prev.d_colorError,  0, batchNeuronBytes, m_stream));
        CUDA_CHECK(cudaMemsetAsync(prev.d_errorGradX,  0, batchNeuronBytes, m_stream));
        CUDA_CHECK(cudaMemsetAsync(prev.d_errorGradY,  0, batchNeuronBytes, m_stream));

        RunBackwardLayerGPU(
            m_cublasHandle, m_stream,
            m_batchSize, curr.numNeurons, prev.numNeurons,
            curr.d_colorError, curr.d_errorGradX, curr.d_errorGradY,
            prev.d_activations, prev.d_gradX, prev.d_gradY,
            curr.d_weights, curr.d_weightsScale,
            curr.d_preActFreq, curr.d_preActScale,
            curr.d_deltaAFreq, curr.d_deltaXFreq, curr.d_deltaYFreq,
            curr.d_deltaAScale, curr.d_deltaXScale, curr.d_deltaYScale,
            prev.d_colorError, prev.d_errorGradX, prev.d_errorGradY,
            curr.d_deltaWeights, curr.d_deltaWeightsScale,
            curr.d_deltaBiases, curr.d_deltaBiasesScale,
            curr.actType, curr.hasDualWeights
        );
        PROFILE_POP();
    }

    // The loop above, at i=1, already computed prev = m_layers[0]'s
    // d_colorError/d_errorGradX/d_errorGradY (zeroed then GEMM-accumulated into,
    // same as every other layer, layer 0 already has these buffers allocated)
    // Scatter that error into the grid's parameter gradients
    if (m_useGridEncoding) {
        GpuLayer& inputLayer = m_layers[0];
        PROFILE_PUSH_COLOR("GridEncodeBackward", 0xFFFFE1BA);
        RunGridEncodeBackwardGPU(
            m_stream, m_batchSize, m_gridEncoder.numLevels, m_gridEncoder.featuresPerLevel,
            m_gridEncoder.d_levelResolutions, m_gridEncoder.d_levelParamOffsets,
            m_gridEncoder.d_grad,
            m_gridEncoder.d_batchPixelX, m_gridEncoder.d_batchPixelY,
            inputLayer.d_colorError, inputLayer.d_errorGradX, inputLayer.d_errorGradY
        );
        PROFILE_POP();
    }
}

void GpuNetwork::ApplyGradientsGPU(engineFloat baseLearningRate)
{
    // Layer 0 has no weights to optimize, start at 1
    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        GpuLayer& layer = m_layers[i];
        
        // Advance the time step for this layer
        layer.adam_t += 1;
        
        // Calculate total elements
        int numWeights = layer.numNeurons * layer.prevNeurons;
        int numBiases = layer.numNeurons;

        // Note: layer.learningRateMultiplier should be handled either inside the kernel or 
        // by scaling the d_learningRate per layer if they vary. Assuming it's 1.0 like now, 
        // or you can easily pass it to the kernel to multiply against *d_learningRate.
        //engineFloat actualLR = baseLearningRate * layer.learningRateMultiplier;
        
        // Update Weights
        RunAdamOptimizerGPU(m_stream, numWeights, layer.d_weights, layer.d_deltaWeights,
                            layer.d_m_weights, layer.d_v_weights, d_learningRate, layer.learningRateMultiplier,
                            layer.adam_t, m_batchSize);

        // Update Biases
        RunAdamOptimizerGPU(m_stream, numBiases, layer.d_biases, layer.d_deltaBiases,
                            layer.d_m_biases, layer.d_v_biases, d_learningRate, layer.learningRateMultiplier,
                            layer.adam_t, m_batchSize);

        // Update Dual Weights (if applicable)
        if (layer.hasDualWeights) {
            RunAdamOptimizerGPU(m_stream, numWeights, layer.d_weightsScale, layer.d_deltaWeightsScale,
                                layer.d_m_weightsScale, layer.d_v_weightsScale, d_learningRate, layer.learningRateMultiplier,
                                layer.adam_t, m_batchSize);

            RunAdamOptimizerGPU(m_stream, numBiases, layer.d_biasesScale, layer.d_deltaBiasesScale,
                                layer.d_m_biasesScale, layer.d_v_biasesScale, d_learningRate, layer.learningRateMultiplier,
                                layer.adam_t, m_batchSize);
        }
    }

    if (m_useGridEncoding) {
        // One time step for the whole encoder (not per-level all levels train
        // together every batch, so they should share the same Adam bias-correction
        // schedule). Each level gets its own RunAdamOptimizerGPU call over its own
        // slice of the concatenated buffers, exactly like weights vs weightsScale
        // above are two separate calls into the same conceptual "layer"
        m_gridEncoder.adam_t += 1;
        using namespace GridEncoding;
        int offset = 0;
        for (int level = 0; level < m_gridEncoder.numLevels; ++level) {
            int numParams = LevelNumParams(level);
            RunAdamOptimizerGPU(m_stream, numParams,
                                m_gridEncoder.d_params + offset, m_gridEncoder.d_grad + offset,
                                m_gridEncoder.d_m + offset, m_gridEncoder.d_v + offset,
                                d_learningRate, m_gridEncoder.learningRateMultiplier,
                                m_gridEncoder.adam_t, m_batchSize);
            offset += numParams;
        }
    }
}

// Kept for debugging purposes
// void GpuNetwork::TransferDeltasToCPU(Network& cpuNetwork)
// {
//     // Grab the CPU array where the Adam optimizer expects to find the gradients
//     auto& storedDeltas = cpuNetwork.GetStoredDeltas();
//
//     // We start at 1 because Layer 0 is just the input passthrough (no weights)
//     for (size_t i = 1; i < m_layers.size(); ++i)
//     {
//         GpuLayer& gpuLayer = m_layers[i];
//         Parameters& deltaParams = storedDeltas[i]; 
//         
//         size_t wSize = gpuLayer.numNeurons * gpuLayer.prevNeurons * sizeof(engineFloat);
//         size_t bSize = gpuLayer.numNeurons * sizeof(engineFloat);
//
//         // Copy gradients from Device (VRAM) directly into the CPU's stored delta arrays
//         CUDA_CHECK(cudaMemcpy(deltaParams.weights.data(), gpuLayer.d_deltaWeights, wSize, cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(deltaParams.biases.data(), gpuLayer.d_deltaBiases, bSize, cudaMemcpyDeviceToHost));
//
//         if (gpuLayer.hasDualWeights) {
//             CUDA_CHECK(cudaMemcpy(deltaParams.weights_scale.data(), gpuLayer.d_deltaWeightsScale, wSize, cudaMemcpyDeviceToHost));
//             CUDA_CHECK(cudaMemcpy(deltaParams.biases_scale.data(), gpuLayer.d_deltaBiasesScale, bSize, cudaMemcpyDeviceToHost));
//         }
//     }
// }