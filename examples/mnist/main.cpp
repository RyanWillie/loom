#include <iostream>

#include "loom/dataloader/dataloader.h"
#include "loom/dataloader/mnist_dataset.h"
#include "loom/device.h"
#include "loom/logger.h"
#include "loom/tensor/tensor.h"

using namespace loom;

int main() {
    // Configure logging
    Logger::setMinLogLevel(LogLevel::INFO);
    Logger::setLogOutput(LogOutput::CONSOLE);
    auto& logger = Logger::getInstance("MNIST");

    logger.info("=== Loom MNIST Example ===");

    // Device configuration
    Device device(DeviceType::CPU);
    logger.info("Running on: {}", device.toString());

    // ========================================================================
    // Load MNIST Dataset
    // ========================================================================
    logger.info("--- Loading MNIST Dataset ---");
    MNISTDataset train_data("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    logger.info("Training samples: {}", train_data.size());
    logger.info("Image size: {} pixels", MNISTDataset::sImageSize);
    logger.info("Number of classes: {}", MNISTDataset::sClasses);

    // ========================================================================
    // Inspect a Single Sample
    // ========================================================================
    logger.info("--- Inspecting First Sample ---");
    auto sample = train_data.get(0);

    logger.info("Input shape:  [{}]", sample.input.size(0));
    logger.info("Target shape: [{}]", sample.target.size(0));
    logger.info("Pixel range:  [{:.3f}, {:.3f}]", sample.input.min().item(),
                sample.input.max().item());

    // Find the digit label
    auto label_acc = sample.target.accessor<float, 1>();
    for (size_t i = 0; i < MNISTDataset::sClasses; ++i) {
        if (label_acc[i] > 0.5f) {
            logger.info("Label: {}", i);
            break;
        }
    }

    // ========================================================================
    // Create DataLoader with Batching
    // ========================================================================
    logger.info("--- Creating DataLoader ---");
    const size_t batch_size = 64;
    const bool shuffle = true;
    DataLoader train_loader(train_data, batch_size, shuffle);

    logger.info("Batch size: {}", batch_size);
    logger.info("Number of batches: {}", train_loader.numBatches());
    logger.info("Shuffle: {}", shuffle ? "enabled" : "disabled");

    // ========================================================================
    // Process First Batch
    // ========================================================================
    logger.info("--- Processing First Batch ---");
    auto batch = train_loader.getBatch(0);

    logger.info("Batch input shape:  [{}, {}]", batch.input.size(0), batch.input.size(1));
    logger.info("Batch target shape: [{}, {}]", batch.target.size(0), batch.target.size(1));
    logger.info("Batch input range:  [{:.3f}, {:.3f}]", batch.input.min().item(),
                batch.input.max().item());

    // ========================================================================
    // Simulate Training Loop (Framework Demo)
    // ========================================================================
    logger.info("--- Simulating Training Loop ---");
    const size_t num_epochs = 2;
    const size_t batches_to_show = 3;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        logger.info("Epoch {}/{}", epoch + 1, num_epochs);

        // In a real training loop, you'd:
        // 1. Forward pass: predictions = model(batch.input)
        // 2. Compute loss: loss = criterion(predictions, batch.target)
        // 3. Backward pass: loss.backward()
        // 4. Update weights: optimizer.step()

        for (size_t batch_idx = 0; batch_idx < std::min(batches_to_show, train_loader.numBatches());
             ++batch_idx) {
            auto batch = train_loader.getBatch(batch_idx);

            // Simulate forward pass - just compute mean pixel value for demo
            float mean_pixel = batch.input.mean().item();

            logger.info("  Batch {}/{}: mean_pixel={:.4f}", batch_idx + 1,
                        train_loader.numBatches(), mean_pixel);
        }

        // Reshuffle for next epoch
        train_loader.reset();
    }

    logger.info("=== MNIST Example Complete ===");
    Logger::shutdown();
    return 0;
}
