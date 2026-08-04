#include <fstream>
#include <vector>
#include <iostream>

struct ParsedData {
    std::vector<std::vector<engineFloat>> inputs;  // Inner vector: 16 floats (4 pixels * 4 channels)
    std::vector<std::vector<engineFloat>> outputs; // Inner vector: 12 floats (3 pixels * 4 channels)
};

bool ParseBinaryData(const char* filename, ParsedData& outData) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open binary file: " << filename << std::endl;
        return false;
    }

    // A block contains 4 input pixels + 3 output pixels = 7 pixels total
    // 7 pixels * 3 bytes per pixel = 28 bytes per iteration loop
    const size_t bytesPerBlock = 21;

    // Read individual raw bytes into a temporary buffer row block
    unsigned char blockBuffer[bytesPerBlock];

    while (file.read(reinterpret_cast<char*>(blockBuffer), bytesPerBlock)) {
        size_t bufferIdx = 0;

        // Process the 4 Input Pixels (12 bytes -> 12 floats)
        std::vector<engineFloat> currentInputs(12);
        for (size_t i = 0; i < 12; ++i) {
            // Read byte, cast to engineFloat, normalize down to 0.0f - 1.0f
            currentInputs[i] = static_cast<engineFloat>(blockBuffer[bufferIdx++]) / 255.0f;
        }
        outData.inputs.push_back(currentInputs);

        // Process the 3 Output Pixels (9 bytes -> 9 floats)
        std::vector<engineFloat> currentOutputs(9);
        for (size_t i = 0; i < 9; ++i) {
            currentOutputs[i] = static_cast<engineFloat>(blockBuffer[bufferIdx++]) / 255.0f;
        }
        outData.outputs.push_back(currentOutputs);
    }

    file.close();
    return true;
}