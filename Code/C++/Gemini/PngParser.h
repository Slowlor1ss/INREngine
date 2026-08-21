#pragma once

#include <algorithm>

#include "IImageParser.h"
// Include STB headers
#include "stb_image.h"
#include "stb_image_write.h"

class PNGParser : public IImageParser {
public:
    ImgParser::Image load(const std::string& filename) override {
        ImgParser::Image img;
        int channels;

        // Force loading as 3 channels (RGB) regardless of whether the PNG has alpha
        unsigned char* raw_data = stbi_load(filename.c_str(), &img.width, &img.height, &channels, 3);

        if (!raw_data) {
            std::cerr << "Error: Could not load PNG file " << filename << "\n";
            return img;
        }

        int total_pixels = img.width * img.height * 3;
        img.data.resize(total_pixels);

        // Normalize 0-255 bytes to 0.0-1.0 floats
        for (int i = 0; i < total_pixels; ++i) {
            img.data[i] = raw_data[i] / 255.0f;
        }

        stbi_image_free(raw_data);
        return img;
    }

    bool parseData(const std::string& filename, ImgParser::ImageParsedData& outData) override {
        ImgParser::Image img = load(filename);
        if (img.width == 0 || img.height == 0) return false;

        outData.width = img.width;
        outData.height = img.height;

        // Reusing the exact same normalization logic
        for (size_t y = 0; y < img.height; y++) {
            for (size_t x = 0; x < img.width; x++) {
                float c1 = img.data[(y * img.width + x) * 3];     
                float c2 = img.data[(y * img.width + x) * 3 + 1]; 
                float c3 = img.data[(y * img.width + x) * 3 + 2]; 

                float normX = (x / float(img.width)) * 2.0f - 1.0f; 
                float normY = (y / float(img.height)) * 2.0f - 1.0f; 

                outData.inputs.push_back({ static_cast<engineFloat>(normX), static_cast<engineFloat>(normY) }); 
                outData.outputs.push_back({ static_cast<engineFloat>(c1), static_cast<engineFloat>(c2), static_cast<engineFloat>(c3) }); 
            }
        }
        return true;
    }

    void save(const std::string& filename, int width, int height, const std::vector<engineFloat>& data) override {
        std::vector<uint8_t> byte_data(width * height * 3);

        // Convert network outputs back to bytes
        for (int i = 0; i < width * height * 3; ++i) {
            byte_data[i] = static_cast<uint8_t>(std::clamp(data[i], 0.0f, 1.0f) * 255.0f); 
        }

        // STB expects stride to be the length of one row in bytes
        int stride_in_bytes = width * 3;

        if (!stbi_write_png(filename.c_str(), width, height, 3, byte_data.data(), stride_in_bytes)) {
            std::cerr << "Error: Failed to save PNG file " << filename << "\n";
        }
    }
};
