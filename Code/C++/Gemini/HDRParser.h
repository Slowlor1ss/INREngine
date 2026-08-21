#pragma once

#include "IImageParser.h"
#include "stb_image.h"
#include "stb_image_write.h"

class HDRParser : public IImageParser {
public:
    ImgParser::Image load(const std::string& filename) override {
        ImgParser::Image img;
        int channels;
        // Use stbi_loadf instead of stbi_load for HDR!
        float* raw_data = stbi_loadf(filename.c_str(), &img.width, &img.height, &channels, 3);
        
        if (!raw_data) return img;

        int total_pixels = img.width * img.height * 3;
        img.data.resize(total_pixels);
        
        // No division by 255.0f needed, it is already floats.
        for (int i = 0; i < total_pixels; ++i) {
            img.data[i] = raw_data[i];
        }

        stbi_image_free(raw_data);
        return img;
    }

    bool parseData(const std::string& filename, ImgParser::ImageParsedData& outData) override {
        ImgParser::Image img = load(filename);
        if (img.width == 0 || img.height == 0) return false;
        
        outData.width = img.width;   
        outData.height = img.height; 

        for (size_t y = 0; y < img.height; y++) { 
            for (size_t x = 0; x < img.width; x++) { 
                float normX = (x / float(img.width)) * 2.0f - 1.0f; 
                float normY = (y / float(img.height)) * 2.0f - 1.0f; 

                outData.inputs.push_back({ static_cast<engineFloat>(normX), static_cast<engineFloat>(normY) }); 
                outData.outputs.push_back({ 
                    static_cast<engineFloat>(img.data[(y * img.width + x) * 3]),     
                    static_cast<engineFloat>(img.data[(y * img.width + x) * 3 + 1]), 
                    static_cast<engineFloat>(img.data[(y * img.width + x) * 3 + 2])  
                });
            }
        }
        return true;
    }

	void save(const std::string& filename, int width, int height, const std::vector<engineFloat>& data) override {
        // Evaluate at compile-time whether engineFloat is exactly 'float'
        if constexpr (std::is_same_v<engineFloat, float>) {
            // Zero-overhead path: directly pass the data
            stbi_write_hdr(filename.c_str(), width, height, 3, data.data());
        } else {
            // This branch is only compiled if engineFloat is NOT float (e.g., double).
            // It warns you at runtime ONLY if this function is actually called.
            std::cerr << "[WARNING] HDRParser: engineFloat is not 'float'. Performing automatic conversion for saving.\n";
            
            // Create a temporary float vector for stbi_write_hdr
            std::vector<float> temp_data(data.begin(), data.end());
            stbi_write_hdr(filename.c_str(), width, height, 3, temp_data.data());
        }
    }
};