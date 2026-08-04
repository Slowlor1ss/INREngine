#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>

// Struct to hold our image data
struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> data; // Stores RGB pixels as floats from 0.0 to 1.0
};

Image loadBMP(const std::string& filename) {
    Image img;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return img;
    }

    // BMP Header is always 54 bytes
    uint8_t header[54];
    if (!file.read(reinterpret_cast<char*>(header), 54)) {
        std::cerr << "Error: File too small to be a BMP\n";
        return img;
    }

    // Check magic numbers 'B' and 'M'
    if (header[0] != 'B' || header[1] != 'M') {
        std::cerr << "Error: Not a valid BMP file\n";
        return img;
    }

    // Safely extract integers using bit-shifting (avoids endianness issues)
    int dataOffset = header[10] | (header[11] << 8) | (header[12] << 16) | (header[13] << 24);
    img.width = header[18] | (header[19] << 8) | (header[20] << 16) | (header[21] << 24);
    img.height = header[22] | (header[23] << 8) | (header[24] << 16) | (header[25] << 24);
    int bpp = header[28] | (header[29] << 8);

    int bytesperPixel = bpp / 8;

    // BMP images are traditionally stored bottom-to-top.
    // If height is negative, it's stored top-to-bottom.
    bool bottomUp = true;
    if (img.height < 0) {
        img.height = -img.height;
        bottomUp = false;
    }

    // Resize our float array to hold width * height * 3 (for R, G, B)
    img.data.resize(img.width * img.height * 3);

    // BMP rows are padded to be a multiple of 4 bytes.
    int row_padded = (img.width * bytesperPixel + 3) & (~3);
    std::vector<uint8_t> row_data(row_padded);

    // Jump to the start of the pixel data
    file.seekg(dataOffset, std::ios::beg);

    for (int i = 0; i < img.height; ++i) {
        file.read(reinterpret_cast<char*>(row_data.data()), row_padded);

        // Calculate the actual row index based on the orientation
        int row = bottomUp ? (img.height - 1 - i) : i;

        for (int j = 0; j < img.width; ++j) {
            // BMPs store pixels in BGR order, not RGB
            uint8_t b = row_data[j * bytesperPixel + 0];
            uint8_t g = row_data[j * bytesperPixel + 1];
            uint8_t r = row_data[j * bytesperPixel + 2];

            // Normalize bytes (0-255) to floats (0.0 - 1.0)
            int pixel_index = (row * img.width + j) * 3;
            img.data[pixel_index + 0] = r / 255.0f;
            img.data[pixel_index + 1] = g / 255.0f;
            img.data[pixel_index + 2] = b / 255.0f;
        }
    }

    return img;
}

struct BMPParsedData {
    int width = 0;
    int height = 0;
    std::vector<std::vector<engineFloat>> inputs;
    std::vector<std::vector<engineFloat>> outputs;
};

bool ParseBMPData(const char* filename, BMPParsedData& outData) {
    Image img = loadBMP(filename);

    // Save dimensions for later
    outData.width = img.width;
    outData.height = img.height;

    for (size_t y = 0; y < img.height; y++)
	{
	    for (size_t x = 0; x < img.width; x++)
	    {
	        float c1 = img.data[(y * img.width + x) * 3];
	        float c2 = img.data[(y * img.width + x) * 3 + 1];
	        float c3 = img.data[(y * img.width + x) * 3 + 2];

	        // Map inputs from [0, 1] to [-1, 1]
	        float normX = (x / float(img.width)) * 2.0f - 1.0f;
	        float normY = (y / float(img.height)) * 2.0f - 1.0f;

	        outData.inputs.push_back({ 
                static_cast<engineFloat>(normX), 
                static_cast<engineFloat>(normY) 
            });
	        outData.outputs.push_back({ 
                static_cast<engineFloat>(c1), 
                static_cast<engineFloat>(c2), 
                static_cast<engineFloat>(c3) 
            });
	    }
	}
    return true;
}

inline engineFloat clamp(const engineFloat v, const engineFloat l, const engineFloat h) {
    if (v > h) return h;
    if (v < l) return l;
    return v;
}

void saveBMP(const std::string& filename, int width, int height, const std::vector<engineFloat>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    int row_padded = (width * 3 + 3) & (~3);
    int fileSize = 54 + row_padded * height;

    // Standard 54-byte BMP header
    uint8_t header[54] = {
        'B','M', // Magic number
        static_cast<uint8_t>(fileSize), static_cast<uint8_t>(fileSize >> 8), static_cast<uint8_t>(fileSize >> 16), static_cast<uint8_t>(fileSize >> 24),
        0,0,0,0,
        54,0,0,0, // Data offset
        40,0,0,0, // Info header size
        static_cast<uint8_t>(width), static_cast<uint8_t>(width >> 8), static_cast<uint8_t>(width >> 16), static_cast<uint8_t>(width >> 24),
        static_cast<uint8_t>(height), static_cast<uint8_t>(height >> 8), static_cast<uint8_t>(height >> 16), static_cast<uint8_t>(height >> 24),
        1,0,      // Planes
        24,0,     // Bits per pixel (24-bit RGB)
        0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
    };
    file.write(reinterpret_cast<char*>(header), 54);

    std::vector<uint8_t> row_data(row_padded, 0);

    // Write pixels (BMPs are naturally written bottom-to-top)
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;

            // NN outputs might slightly exceed bounds; clamp them safely
            engineFloat r = clamp(data[idx + 0], 0, 1);
            engineFloat g = clamp(data[idx + 1], 0, 1);
            engineFloat b = clamp(data[idx + 2], 0, 1);

            // BMPs store pixels in BGR order
            row_data[x * 3 + 0] = static_cast<uint8_t>(b * 255.0f);
            row_data[x * 3 + 1] = static_cast<uint8_t>(g * 255.0f);
            row_data[x * 3 + 2] = static_cast<uint8_t>(r * 255.0f);
        }
        file.write(reinterpret_cast<char*>(row_data.data()), row_padded);
    }
}