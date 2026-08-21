#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>

// TODO: we dont use this file anymore

// Helper function to swap endianness for 32-bit integers
inline uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
        ((val << 8) & 0x00FF0000) |
        ((val >> 8) & 0x0000FF00) |
        ((val >> 24) & 0x000000FF);
}

// Function to read MNIST images into a vector of vectors
inline std::vector<std::vector<float>> read_mnist_images(const std::string& filepath, bool normalize = true) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    // Read headers
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

    // Swap to host endianness
    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number.");
    }

    size_t pixels_per_image = num_rows * num_cols;
    size_t total_pixels = num_images * pixels_per_image;

    // Read all raw uint8_t data into a flat buffer first for performance
    std::vector<uint8_t> raw_data(total_pixels);
    file.read(reinterpret_cast<char*>(raw_data.data()), total_pixels);

    // Initialize the 2D vector: num_images vectors, each of size pixels_per_image
    std::vector<std::vector<float>> images(num_images, std::vector<float>(pixels_per_image));

    // Populate the 2D vector
    for (size_t i = 0; i < num_images; ++i) {
        for (size_t j = 0; j < pixels_per_image; ++j) {
            size_t raw_index = i * pixels_per_image + j;
            if (normalize) {
                images[i][j] = raw_data[raw_index] / 255.0f; // Normalize to [0.0, 1.0]
            }
            else {
                images[i][j] = static_cast<float>(raw_data[raw_index]);
            }
        }
    }

    return images;
}

// Function to read MNIST labels and one-hot encode them
std::vector<std::vector<float>> read_mnist_labels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t magic_number = 0;
    uint32_t num_items = 0;

    // Read headers
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));

    // Swap to host endianness
    magic_number = swap_endian(magic_number);
    num_items = swap_endian(num_items);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number.");
    }

    // Read the raw label data (0-9) into a temporary buffer
    std::vector<uint8_t> raw_labels(num_items);
    file.read(reinterpret_cast<char*>(raw_labels.data()), num_items);

    // Create the one-hot encoded vector
    // Initialize num_items arrays, each with 10 floats defaulted to 0.0f
    std::vector<std::vector<float>> one_hot_labels(num_items, std::vector<float>(10, 0.0f));

    for (size_t i = 0; i < num_items; ++i) {
        uint8_t label_val = raw_labels[i];

        // Safety check to ensure the digit is valid (0 through 9)
        if (label_val < 10) {
            one_hot_labels[i][label_val] = 1.0f;
        }
        else {
            throw std::runtime_error("Encountered a label outside the 0-9 range.");
        }
    }

    return one_hot_labels;
}