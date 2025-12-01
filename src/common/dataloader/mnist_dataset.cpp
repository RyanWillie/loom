#include "common/dataloader/mnist_dataset.h"

#include <cstdint>
#include <cstdio>
#include <fstream>

#include "common/logger.h"

loom::MNISTDataset::MNISTDataset(const std::string& data_path, const std::string& label_path)
    : mImages(Tensor::zeros({0, sImageSize}, DType::FLOAT32)),
      mLabels(Tensor::zeros({0, sClasses}, DType::FLOAT32)),
      mSize(0) {
    loadImages(data_path);
    loadLabels(label_path);
    mSize = mImages.shape()[0];
}

size_t loom::MNISTDataset::size() const {
    return mSize;
}

loom::Sample loom::MNISTDataset::get(size_t index) const {
    return {mImages.slice(0, index, index + 1).squeeze(),
            mLabels.slice(0, index, index + 1).squeeze()};
}

uint32_t loom::MNISTDataset::readBigEndianUint32(std::istream& stream) {
    uint8_t bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), sizeof(uint8_t) * 4);
    // Big-endian: first byte is most significant
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | (static_cast<uint32_t>(bytes[3]));
}

void loom::MNISTDataset::loadImages(const std::string& path) {
    auto& logger = Logger::getInstance("MNIST", LogLevel::DEBUG);
    logger.debug("Loading images from {}", path);
    // 1. Open file in binary mode
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open images file: " + path);
    }

    // 2. Read header
    uint32_t magic = readBigEndianUint32(file);
    logger.debug("Magic number: {}", magic);
    if (magic != 0x00000803) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t numImages = readBigEndianUint32(file);
    logger.debug("Number of images: {}", numImages);
    // 3. Allocate tensor [numImages, 784]
    mImages = Tensor::zeros({numImages, sImageSize}, DType::FLOAT32);
    logger.debug("Allocated tensor [{}x{}]", numImages, sImageSize);
    // 4. Read pixel data and normalize
    auto accessor = mImages.accessor<float, 2>();

    for (uint32_t i = 0; i < numImages; ++i) {
        for (uint32_t j = 0; j < sImageSize; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);

            // Normalize: [0, 255] → [0.0, 1.0]
            accessor[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
}

void loom::MNISTDataset::loadLabels(const std::string& path) {
    auto& logger = Logger::getInstance("MNIST", LogLevel::DEBUG);
    logger.debug("Loading labels from {}", path);
    // 1. Open file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open labels file: " + path);
    }

    // 2. Read header
    uint32_t magic = readBigEndianUint32(file);
    logger.debug("Magic number: {}", magic);
    if (magic != 0x00000801) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t numLabels = readBigEndianUint32(file);
    logger.debug("Number of labels: {}", numLabels);
    // 3. Allocate one-hot tensor [numLabels, 10]
    mLabels = Tensor::zeros({numLabels, sClasses}, DType::FLOAT32);
    logger.debug("Allocated tensor [{}x{}]", numLabels, sClasses);
    // 4. Read labels and one-hot encode
    auto accessor = mLabels.accessor<float, 2>();

    for (uint32_t i = 0; i < numLabels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);

        // One-hot: set the label index to 1.0
        accessor[i][label] = 1.0f;
    }
}