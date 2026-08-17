#pragma once
#include <vector>
#include <cstdint>
#define NOMINMAX
#include <windows.h>

#include "Types.h"

class ImageWindow {
public:
    ImageWindow(int width, int height);
    ~ImageWindow();

    // Takes an array of RGB floats (0.0 to 1.0) and draws it
    void Update(const std::vector<engineFloat>& rgbData);
    
    // Call this in your main loop so the window doesn't freeze
    void ProcessMessages(); 

private:
    HWND m_hwnd;
    int m_width;
    int m_height;
    std::vector<uint8_t> m_displayBuffer;
    BITMAPINFO m_bmi;
};