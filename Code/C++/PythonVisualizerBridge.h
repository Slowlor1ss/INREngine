#pragma once
#include "VizProtocol.h"
#include "Types.h"
#include <windows.h>
#include <string>
#include <vector>

// "Bridge" that pushes render frames into named
// shared memory so we can use python to render
class PythonVisualizerBridge
{
public:
    // renderWidth/Height: dimentions of the image we create
    // targetWidth/Height: dimensions of reference image (can pass 0,0 if none)
    // channels: 3 for RGB etc...
    // tag: shared memory namespace defaults to "INR_Default" to match python/viz/main.py default
    PythonVisualizerBridge(int renderWidth, int renderHeight, int targetWidth, int targetHeight,
                            int channels, bool autoClose = true, const std::string& tag = "INR_Default");
    ~PythonVisualizerBridge();

    PythonVisualizerBridge(const PythonVisualizerBridge&) = delete;
    PythonVisualizerBridge& operator=(const PythonVisualizerBridge&) = delete;
    
    // Not seqlock protected and thus should be written once before any reader could touch it
    void PushReferenceFrame(const std::vector<engineFloat>& rgbData);
    
    // Non blocking and seqlock protected
    void PushLiveFrame(const std::vector<engineFloat>& rgbData, size_t epoch, engineFloat cost, engineFloat learningRate);

    // Can skip and launch py process manually
    bool LaunchViewerProcess(const std::wstring& pythonExe = L"python", const std::wstring& scriptPath = L"Code/scripts/viz/main.py");

    bool IsValid() const { return m_header != nullptr && m_currentFrame != nullptr; }

private:
    static HANDLE CreateMapping(const std::wstring& name, size_t bytes);

private:
    std::string m_tag;
    int m_renderWidth = 0, m_renderHeight = 0;
    int m_targetWidth = 0, m_targetHeight = 0;
    int m_channels = 3;

    HANDLE m_headerMapping = nullptr;
    HANDLE m_currentMapping = nullptr;
    HANDLE m_targetMapping = nullptr;

    VizSharedHeader* m_header = nullptr;
    engineFloat* m_currentFrame = nullptr;
    engineFloat* m_targetFrame = nullptr;

    PROCESS_INFORMATION m_viewerProcess{};
    bool m_viewerLaunched = false;
    bool m_autoClose = false;
};
