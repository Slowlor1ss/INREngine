#include "PythonVisualizerBridge.h"
#include <atomic>
#include <iostream>

namespace
{
    std::wstring Widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }
}

HANDLE PythonVisualizerBridge::CreateMapping(const std::wstring& name, size_t bytes)
{
    // https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-createfilemappingw
    const HANDLE h = CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        // high-order DWORD of the maximum size of the file mapping object
        static_cast<DWORD>((static_cast<uint64_t>(bytes) >> 32) & 0xFFFFFFFFu), 
        // low-order DWORD of the maximum size of the file mapping object
        static_cast<DWORD>(bytes & 0xFFFFFFFFu),
        // The name of the file mapping object
        // If this parameter matches the name of an existing mapping object, 
        // the function requests access to the object with the protection that flProtect specifies
        name.c_str());

    if (!h)
    {
        std::wcerr << L"[PythonVisualizerBridge] CreateFileMappingW failed for '" << name
                    << L"', Last Error: " << GetLastError() << L"\n";
    }
    return h;
}

PythonVisualizerBridge::PythonVisualizerBridge(int renderWidth, int renderHeight, int targetWidth, int targetHeight,
                                                int channels, bool autoclose, const std::string& tag)
    : m_tag(tag), m_renderWidth(renderWidth), m_renderHeight(renderHeight)
    , m_targetWidth(targetWidth), m_targetHeight(targetHeight), m_channels(channels)
    , m_autoClose(autoclose)
{
    const std::wstring wtag = Widen(tag);

    const size_t currentBytes = static_cast<size_t>(renderWidth) * renderHeight * channels * sizeof(engineFloat);
    const size_t targetBytes = static_cast<size_t>(targetWidth) * targetHeight * channels * sizeof(engineFloat);

    // https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-createfilemappingwl
    // The name can have a "Global" or "Local" prefix to explicitly create the object in the global or session namespace
    m_headerMapping = CreateMapping(L"Local\\" + wtag + L"_Header", sizeof(VizSharedHeader));
    m_currentMapping = CreateMapping(L"Local\\" + wtag + L"_Current", currentBytes);
    if (targetBytes > 0)
        m_targetMapping = CreateMapping(L"Local\\" + wtag + L"_Target", targetBytes);

    if (m_headerMapping)
        m_header = static_cast<VizSharedHeader*>(MapViewOfFile(m_headerMapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(VizSharedHeader)));
    if (m_currentMapping)
        m_currentFrame = static_cast<engineFloat*>(MapViewOfFile(m_currentMapping, FILE_MAP_ALL_ACCESS, 0, 0, currentBytes));
    if (m_targetMapping)
        m_targetFrame = static_cast<engineFloat*>(MapViewOfFile(m_targetMapping, FILE_MAP_ALL_ACCESS, 0, 0, targetBytes));

    if (!IsValid())
    {
        std::cerr << "[PythonVisualizerBridge] Failed to set up shared memory (tag='" << tag << "'). "
                     "The live Python viewer will be unavailable; training itself is unaffected.\n";
        return;
    }

    // Explicit zero-init: also resets stale state from a previous run that reused this tag
    // (CreateFileMapping zero-fills fresh pages, but a *reused* mapping from a still-running
    // old process would not be, so just in case)
    *m_header = VizSharedHeader{};
    m_header->width = static_cast<uint32_t>(renderWidth);
    m_header->height = static_cast<uint32_t>(renderHeight);
    m_header->channels = static_cast<uint32_t>(channels);
    m_header->targetWidth = static_cast<uint32_t>(targetWidth);
    m_header->targetHeight = static_cast<uint32_t>(targetHeight);
    m_header->hasTarget = targetBytes > 0 ? 1u : 0u;
    m_header->running = 1;

    std::cout << "[PythonVisualizerBridge] Shared memory ready (tag='" << tag << "', "
              << renderWidth << "x" << renderHeight << "). Run `python python/viz/main.py --tag " << tag << "` to view.\n";
}

PythonVisualizerBridge::~PythonVisualizerBridge()
{
    if (m_header)
    {
        m_header->running = 0; // Tells the Python viewer we've stopped, distinct from "no new frames yet"
        UnmapViewOfFile(m_header);
    }
    if (m_currentFrame) UnmapViewOfFile(m_currentFrame);
    if (m_targetFrame) UnmapViewOfFile(m_targetFrame);

    if (m_headerMapping) CloseHandle(m_headerMapping);
    if (m_currentMapping) CloseHandle(m_currentMapping);
    if (m_targetMapping) CloseHandle(m_targetMapping);

    if (m_viewerLaunched)
    {
        if (m_autoClose)
            TerminateProcess(m_viewerProcess.hProcess, 0);
        
        CloseHandle(m_viewerProcess.hProcess);
        CloseHandle(m_viewerProcess.hThread);
    }
}

void PythonVisualizerBridge::PushReferenceFrame(const std::vector<engineFloat>& rgbData)
{
    if (!m_targetFrame || !m_header) return;

    const size_t floatCount = static_cast<size_t>(m_targetWidth) * m_targetHeight * m_channels;
    if (rgbData.size() < floatCount)
    {
        std::cerr << "[PythonVisualizerBridge] PushReferenceFrame: data too small ("
                  << rgbData.size() << " < " << floatCount << "), skipping.\n";
        return;
    }
    memcpy(m_targetFrame, rgbData.data(), floatCount * sizeof(engineFloat));
}

void PythonVisualizerBridge::PushLiveFrame(const std::vector<engineFloat>& rgbData, size_t epoch, engineFloat cost, engineFloat learningRate)
{
    if (!m_currentFrame || !m_header) return;

    const size_t floatCount = static_cast<size_t>(m_renderWidth) * m_renderHeight * m_channels;
    if (rgbData.size() < floatCount) return;

    // Seqlock write: bump to odd(write in progress), copy, bump to even(stable)
    // A reader that samples the counter as odd (or sees it change across its read) just
    // retries, see shared_memory_bridge.py read_current_frame()
    std::atomic_ref<uint32_t> counter(m_header->frameCounter);
    counter.fetch_add(1, std::memory_order_acquire);
    memcpy(m_currentFrame, rgbData.data(), floatCount * sizeof(engineFloat));
    m_header->epoch = epoch;
    m_header->cost = cost;
    m_header->learningRate = learningRate;
    counter.fetch_add(1, std::memory_order_release);
}

bool PythonVisualizerBridge::LaunchViewerProcess(const std::wstring& pythonExe, const std::wstring& scriptPath)
{
    std::wstring cmd = pythonExe + L" \"" + scriptPath + L"\" --tag " + Widen(m_tag); //+ L" --quiet";

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    ZeroMemory(&m_viewerProcess, sizeof(m_viewerProcess));
    
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                              /*CREATE_NEW_CONSOLE*/ 0, nullptr, nullptr, &si, &m_viewerProcess);
    m_viewerLaunched = (ok != 0);
    if (!ok)
        std::cerr << "[PythonVisualizerBridge] Failed to launch viewer process, Last Error: " << GetLastError() << "\n";
    return m_viewerLaunched;
}
