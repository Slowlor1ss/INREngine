#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

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

    // We are only handling standard 24-bit BMPs for simplicity
    if (bpp != 24) {
        std::cerr << "Error: Only 24-bit BMPs are supported\n";
        return img;
    }

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
    int row_padded = (img.width * 3 + 3) & (~3);
    std::vector<uint8_t> row_data(row_padded);

    // Jump to the start of the pixel data
    file.seekg(dataOffset, std::ios::beg);

    for (int i = 0; i < img.height; ++i) {
        file.read(reinterpret_cast<char*>(row_data.data()), row_padded);

        // Calculate the actual row index based on the orientation
        int row = bottomUp ? (img.height - 1 - i) : i;

        for (int j = 0; j < img.width; ++j) {
            // BMPs store pixels in BGR order, not RGB
            uint8_t b = row_data[j * 3 + 0];
            uint8_t g = row_data[j * 3 + 1];
            uint8_t r = row_data[j * 3 + 2];

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
    std::vector<std::vector<float>> inputs; 
    std::vector<std::vector<float>> outputs;
};

bool ParseBMPData(const char* filename, BMPParsedData& outData) {

    Image img = loadBMP(filename);

    for (size_t y = 0; y < img.height; y++)
    {
        for (size_t x = 0; x < img.width; x++)
        {
            float c1 = img.data[(y * img.width + x) * 3];
            float c2 = img.data[(y * img.width + x) * 3 + 1];
            float c3 = img.data[(y * img.width + x) * 3 + 2];

            outData.inputs.push_back({ x / float(img.width), y / float(img.height) });
            outData.outputs.push_back({ c1,c2,c3 });
        }
    }

    return true;
}
