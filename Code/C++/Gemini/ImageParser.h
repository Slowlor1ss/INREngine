#pragma once

#include <iostream>
#include <fstream>
#include <print>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <memory>

#include "IImageParser.h"
#include "BMPParser.h"
#include "HDRParser.h"
#include "JPEGParser.h"
#include "PngParser.h"
#include "TGAParser.h"
#include "../Types.h"

class ImageParser {
private:
    static std::string getExtension(const std::string& filename) {
        size_t dotPos = filename.find_last_of('.');
        if (dotPos == std::string::npos) return "";
        
        std::string ext = filename.substr(dotPos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext;
    }

	static std::unique_ptr<IImageParser> getParserForFile(const std::string& filename) {
        std::string ext = getExtension(filename);
        
        if (ext == "bmp") {
            return std::make_unique<BMPParser>();
        } else if (ext == "png") {
            return std::make_unique<PNGParser>();
        } else if (ext == "jpg" || ext == "jpeg") {
            return std::make_unique<JPEGParser>();
        } else if (ext == "tga") {
            return std::make_unique<TGAParser>();
        } else if (ext == "hdr") {
            return std::make_unique<HDRParser>();
        }
        
        std::cerr << "Error: Unsupported file format '." << ext << "'\n";
        return nullptr;
    }

public:
    static ImgParser::Image Load(const std::string& filename) {
        auto parser = getParserForFile(filename);
        if (parser) return parser->load(filename);
        return ImgParser::Image(); 
    }

    static bool ParseData(const std::string& filename, ImgParser::ImageParsedData& outData) {
        auto parser = getParserForFile(filename);
        if (parser) return parser->parseData(filename, outData);
#ifndef _TRAINING
            __debugbreak(); // This is fatal
#endif
        return false;
    }

    static void Save(const std::string& filename, int width, int height, const std::vector<engineFloat>& data) {
        auto parser = getParserForFile(filename);
        if (parser) parser->save(filename, width, height, data);
        
        if (errno != 0) {
            char err_msg[256];
            strerror_s(err_msg, sizeof(err_msg), errno);
            std::println(stderr, "File system error: {}", err_msg);
        }
    }
};