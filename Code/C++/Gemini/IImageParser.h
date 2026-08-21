#pragma once
#include <string>

namespace  ImgParser
{
	// Struct to hold our image data
	struct Image {
	    int width = 0;
	    int height = 0;
	    std::vector<float> data; // Stores RGB pixels as floats from 0.0 to 1.0
	};

	struct ImageParsedData {
	    int width = 0;
	    int height = 0;
	    std::vector<std::vector<engineFloat>> inputs;
	    std::vector<std::vector<engineFloat>> outputs;
	};
}

class IImageParser {
public:
    virtual ~IImageParser() = default;
    
    virtual ImgParser::Image load(const std::string& filename) = 0;
    virtual bool parseData(const std::string& filename, ImgParser::ImageParsedData& outData) = 0;
    virtual void save(const std::string& filename, int width, int height, const std::vector<engineFloat>& data) = 0;
};