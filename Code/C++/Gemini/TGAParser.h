#pragma once

#include "IImageParser.h"
#include "stb_image.h"
#include "stb_image_write.h"

class TGAParser : public IImageParser {
public:
    ImgParser::Image load(const std::string& filename) override {
        ImgParser::Image img;
        int channels;
        unsigned char* raw_data = stbi_load(filename.c_str(), &img.width, &img.height, &channels, 3);
        
        if (!raw_data) return img;

        int total_pixels = img.width * img.height * 3;
        img.data.resize(total_pixels);
        for (int i = 0; i < total_pixels; ++i) img.data[i] = raw_data[i] / 255.0f;

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
        std::vector<uint8_t> byte_data(width * height * 3);
        for (int i = 0; i < width * height * 3; ++i) {
            byte_data[i] = static_cast<uint8_t>(std::clamp(data[i], (engineFloat)0.0f, (engineFloat)1.0f) * 255.0f); 
        }
        stbi_write_tga(filename.c_str(), width, height, 3, byte_data.data());
    }
};