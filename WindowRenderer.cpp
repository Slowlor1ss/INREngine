#include "WindowRenderer.h"
#include <iostream>

// Standard Win32 Window Procedure
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

ImageWindow::ImageWindow(int width, int height) : m_width(width), m_height(height) {
    const wchar_t* CLASS_NAME = L"NNRendererClass";
    
    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = CLASS_NAME;
    RegisterClass(&wc);

    // Calculate window size based on image size
    RECT rect = { 0, 0, width, height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    m_hwnd = CreateWindowEx(
        0, CLASS_NAME, L"Network Live View",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
        NULL, NULL, wc.hInstance, NULL
    );

    ShowWindow(m_hwnd, SW_SHOW);

    // Setup BITMAPINFO for 32-bit BGRA (This avoids complex row padding math)
    ZeroMemory(&m_bmi, sizeof(m_bmi));
    m_bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    m_bmi.bmiHeader.biWidth = width;
    m_bmi.bmiHeader.biHeight = -height; // Negative means origin is top-left
    m_bmi.bmiHeader.biPlanes = 1;
    m_bmi.bmiHeader.biBitCount = 32;
    m_bmi.bmiHeader.biCompression = BI_RGB;

    // Buffer to hold 4 bytes (BGRA) per pixel
    m_displayBuffer.resize(width * height * 4, 0);
}

ImageWindow::~ImageWindow() {
    if (m_hwnd) {
        DestroyWindow(m_hwnd);
    }
}

void ImageWindow::Update(const std::vector<engineFloat>& rgbData) {
    for (int i = 0; i < m_width * m_height; ++i) {
        engineFloat r = rgbData[i * 3 + 0];
        engineFloat g = rgbData[i * 3 + 1];
        engineFloat b = rgbData[i * 3 + 2];

        // Clamp values safely before casting to uint8_t
        r = (r > 1.0f) ? 1.0f : ((r < 0.0f) ? 0.0f : r);
        g = (g > 1.0f) ? 1.0f : ((g < 0.0f) ? 0.0f : g);
        b = (b > 1.0f) ? 1.0f : ((b < 0.0f) ? 0.0f : b);

        // Windows GDI expects BGRA byte order
        m_displayBuffer[i * 4 + 0] = static_cast<uint8_t>(b * 255.0f);
        m_displayBuffer[i * 4 + 1] = static_cast<uint8_t>(g * 255.0f);
        m_displayBuffer[i * 4 + 2] = static_cast<uint8_t>(r * 255.0f);
        m_displayBuffer[i * 4 + 3] = 255; // Alpha channel (opaque)
    }

    // Paint to the screen immediately
    const HDC hdc = GetDC(m_hwnd);
    SetDIBitsToDevice(hdc, 0, 0, m_width, m_height, 0, 0, 0, m_height, 
                      m_displayBuffer.data(), &m_bmi, DIB_RGB_COLORS);
    ReleaseDC(m_hwnd, hdc);
}

void ImageWindow::ProcessMessages() {
    MSG msg;
    // Process all pending Windows messages so the window doesn't freeze
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}