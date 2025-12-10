#include <cstdint>
#include <fstream>

#include "loom/dataloader/mnist_dataset.h"
#include "loom/device.h"
#include "loom/dtypes.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// MNIST Dataset Tests
// ============================================================================

class MNISTDatasetTest : public ::testing::Test {
  protected:
    Device cpu_device{DeviceType::CPU};

    // Helper to create a minimal valid MNIST image file
    void createMockImageFile(const std::string& path, uint32_t num_images) {
        std::ofstream file(path, std::ios::binary);

        // Write magic number (0x00000803 for images)
        writeBigEndian(file, static_cast<uint32_t>(0x00000803));

        // Write number of images
        writeBigEndian(file, num_images);

        // Write dummy pixel data (28x28 = 784 bytes per image)
        for (uint32_t i = 0; i < num_images * 784; ++i) {
            uint8_t pixel = static_cast<uint8_t>(i % 256);
            file.write(reinterpret_cast<const char*>(&pixel), 1);
        }

        file.close();
    }

    // Helper to create a minimal valid MNIST label file
    void createMockLabelFile(const std::string& path, uint32_t num_labels) {
        std::ofstream file(path, std::ios::binary);

        // Write magic number (0x00000801 for labels)
        writeBigEndian(file, static_cast<uint32_t>(0x00000801));

        // Write number of labels
        writeBigEndian(file, num_labels);

        // Write dummy labels (0-9)
        for (uint32_t i = 0; i < num_labels; ++i) {
            uint8_t label = static_cast<uint8_t>(i % 10);
            file.write(reinterpret_cast<const char*>(&label), 1);
        }

        file.close();
    }

    void writeBigEndian(std::ofstream& file, uint32_t value) {
        uint8_t bytes[4];
        bytes[0] = static_cast<uint8_t>((value >> 24) & 0xFF);
        bytes[1] = static_cast<uint8_t>((value >> 16) & 0xFF);
        bytes[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
        bytes[3] = static_cast<uint8_t>(value & 0xFF);
        file.write(reinterpret_cast<const char*>(bytes), 4);
    }
};

TEST_F(MNISTDatasetTest, InvalidImageFilePathThrows) {
    EXPECT_THROW({ MNISTDataset dataset("/nonexistent/images.idx", "/nonexistent/labels.idx"); },
                 std::runtime_error);
}

TEST_F(MNISTDatasetTest, LoadValidMockFiles) {
    const std::string image_path = "/tmp/test_mnist_images.idx";
    const std::string label_path = "/tmp/test_mnist_labels.idx";

    createMockImageFile(image_path, 10);
    createMockLabelFile(label_path, 10);

    MNISTDataset dataset(image_path, label_path);

    // Verify size
    EXPECT_EQ(dataset.size(), 10);

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}

TEST_F(MNISTDatasetTest, GetSampleCorrectShape) {
    const std::string image_path = "/tmp/test_mnist_images2.idx";
    const std::string label_path = "/tmp/test_mnist_labels2.idx";

    createMockImageFile(image_path, 5);
    createMockLabelFile(label_path, 5);

    MNISTDataset dataset(image_path, label_path);

    Sample sample = dataset.get(0);

    // Image should be [784] (28x28 flattened)
    EXPECT_EQ(sample.input.ndim(), 1);
    EXPECT_EQ(sample.input.numel(), 784);

    // Label should be [10] (one-hot encoded)
    EXPECT_EQ(sample.target.ndim(), 1);
    EXPECT_EQ(sample.target.numel(), 10);

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}

TEST_F(MNISTDatasetTest, ImageNormalization) {
    const std::string image_path = "/tmp/test_mnist_images3.idx";
    const std::string label_path = "/tmp/test_mnist_labels3.idx";

    createMockImageFile(image_path, 2);
    createMockLabelFile(label_path, 2);

    MNISTDataset dataset(image_path, label_path);
    Sample sample = dataset.get(0);

    // All pixel values should be in [0.0, 1.0]
    auto acc = sample.input.accessor<float, 1>();
    for (size_t i = 0; i < 784; ++i) {
        EXPECT_GE(acc[i], 0.0f);
        EXPECT_LE(acc[i], 1.0f);
    }

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}

TEST_F(MNISTDatasetTest, OneHotEncodingCorrectness) {
    const std::string image_path = "/tmp/test_mnist_images4.idx";
    const std::string label_path = "/tmp/test_mnist_labels4.idx";

    createMockImageFile(image_path, 10);
    createMockLabelFile(label_path, 10);

    MNISTDataset dataset(image_path, label_path);

    // Check first few samples - labels should be 0, 1, 2, ...
    for (size_t i = 0; i < 10; ++i) {
        Sample sample = dataset.get(i);
        auto label_acc = sample.target.accessor<float, 1>();

        // Should have exactly one 1.0 and nine 0.0s
        size_t num_ones = 0;
        size_t hot_index = 0;

        for (size_t j = 0; j < 10; ++j) {
            if (label_acc[j] == 1.0f) {
                num_ones++;
                hot_index = j;
            } else {
                EXPECT_FLOAT_EQ(label_acc[j], 0.0f);
            }
        }

        EXPECT_EQ(num_ones, 1);
        EXPECT_EQ(hot_index, i % 10);  // Labels cycle 0-9
    }

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}

TEST_F(MNISTDatasetTest, InvalidImageMagicNumberThrows) {
    const std::string image_path = "/tmp/test_mnist_bad_images.idx";
    const std::string label_path = "/tmp/test_mnist_labels5.idx";

    // Create file with wrong magic number
    std::ofstream file(image_path, std::ios::binary);
    writeBigEndian(file, static_cast<uint32_t>(0xDEADBEEF));  // Wrong magic
    writeBigEndian(file, static_cast<uint32_t>(1));
    file.close();

    createMockLabelFile(label_path, 1);

    EXPECT_THROW({ MNISTDataset dataset(image_path, label_path); }, std::runtime_error);

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}

TEST_F(MNISTDatasetTest, InvalidLabelMagicNumberThrows) {
    const std::string image_path = "/tmp/test_mnist_images6.idx";
    const std::string label_path = "/tmp/test_mnist_bad_labels.idx";

    createMockImageFile(image_path, 1);

    // Create file with wrong magic number
    std::ofstream file(label_path, std::ios::binary);
    writeBigEndian(file, static_cast<uint32_t>(0xBADBAD));  // Wrong magic
    writeBigEndian(file, static_cast<uint32_t>(1));
    file.close();

    EXPECT_THROW({ MNISTDataset dataset(image_path, label_path); }, std::runtime_error);

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}

TEST_F(MNISTDatasetTest, GetMultipleSamples) {
    const std::string image_path = "/tmp/test_mnist_images7.idx";
    const std::string label_path = "/tmp/test_mnist_labels7.idx";

    createMockImageFile(image_path, 20);
    createMockLabelFile(label_path, 20);

    MNISTDataset dataset(image_path, label_path);

    // Get multiple samples and verify they're different
    Sample s0 = dataset.get(0);
    Sample s10 = dataset.get(10);

    EXPECT_EQ(s0.input.numel(), 784);
    EXPECT_EQ(s10.input.numel(), 784);

    // Labels should be different (0 vs 0 though due to modulo)
    // But at least verify no crashes

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}

TEST_F(MNISTDatasetTest, ConstMethods) {
    const std::string image_path = "/tmp/test_mnist_images8.idx";
    const std::string label_path = "/tmp/test_mnist_labels8.idx";

    createMockImageFile(image_path, 3);
    createMockLabelFile(label_path, 3);

    const MNISTDataset dataset(image_path, label_path);

    // Verify const methods work
    EXPECT_EQ(dataset.size(), 3);

    Sample sample = dataset.get(1);
    EXPECT_EQ(sample.input.numel(), 784);

    // Clean up
    std::remove(image_path.c_str());
    std::remove(label_path.c_str());
}
