#include <algorithm>
#include <iomanip>
#include <iostream>

#include "loom/dataloader/dataloader.h"
#include "loom/dataloader/mnist_dataset.h"
#include "loom/device.h"
#include "loom/logger.h"
#include "loom/nn/activation.h"
#include "loom/nn/linear.h"
#include "loom/nn/loss.h"
#include "loom/nn/sequential.h"
#include "loom/nn/trainer.h"
#include "loom/optim/sgd.h"
#include "loom/tensor/tensor.h"

using namespace loom;

int main() {
    // Configure logging
    Logger::setMinLogLevel(LogLevel::INFO);
    Logger::setLogOutput(LogOutput::CONSOLE);
    auto& logger = Logger::getInstance("MNIST");

    logger.info("=== Loom MNIST Training Example ===");

    // Device configuration
    Device device(DeviceType::CPU);
    logger.info("Running on: {}", device.toString());

    // ========================================================================
    // Load MNIST Dataset
    // ========================================================================
    logger.info("\n--- Loading MNIST Dataset ---");
    MNISTDataset train_data("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    MNISTDataset test_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

    logger.info("Training samples: {}", train_data.size());
    logger.info("Test samples: {}", test_data.size());
    logger.info("Input dimension: {} pixels", MNISTDataset::sImageSize);
    logger.info("Output classes: {}", MNISTDataset::sClasses);

    // ========================================================================
    // Create DataLoaders
    // ========================================================================
    const size_t batch_size = 64;
    DataLoader train_loader(train_data, batch_size, /*shuffle=*/true);
    DataLoader test_loader(test_data, batch_size, /*shuffle=*/false);

    logger.info("Batch size: {}", batch_size);
    logger.info("Training batches: {}", train_loader.numBatches());
    logger.info("Test batches: {}", test_loader.numBatches());

    // ========================================================================
    // Build Neural Network Model
    // ========================================================================
    logger.info("--- Building Model ---");

    // Create model using Sequential for clean composition
    auto model =
        std::make_shared<nn::Sequential>(std::initializer_list<std::shared_ptr<nn::Module>>{
            std::make_shared<nn::Linear>(784, 128), std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(128, 10)});

    logger.info("Model architecture:");
    logger.info("  Linear(784 -> 128)");
    logger.info("  ReLU()");
    logger.info("  Linear(128 -> 10)");

    // Collect parameters from the entire model
    auto params = model->parameters();
    logger.info("Total parameters: {}", params.size());

    // ========================================================================
    // Setup Loss Function and Optimizer
    // ========================================================================
    logger.info("--- Setup Training Components ---");

    auto criterion = std::make_shared<nn::CrossEntropyLoss>();
    logger.info("Loss function: CrossEntropyLoss");

    const double learning_rate = 0.01;
    auto optimizer = std::make_shared<optim::SGD>(params, learning_rate);
    logger.info("Optimizer: SGD (lr={:.4f})", learning_rate);

    // ========================================================================
    // Training with Trainer
    // ========================================================================
    logger.info("--- Starting Training ---");
    const size_t num_epochs = 5;
    const size_t log_interval = 100;

    // Create trainer with default ConsoleMonitor
    auto trainer = std::make_shared<nn::Trainer>(model, optimizer, criterion);

    // Train the model
    const auto& history = trainer->train(train_loader, test_loader, num_epochs, log_interval);

    // ========================================================================
    // Sample Predictions from Test Set
    // ========================================================================
    logger.info("--- Sample Predictions (Test Set) ---");

    for (size_t i = 0; i < 5; ++i) {
        auto sample = test_data.get(i);

        // Inference: forward pass with batch dimension [784] -> [1, 784]
        Tensor input = sample.input.unsqueeze(0);
        Tensor output = model->forward(input);

        // Get predictions (argmax)
        auto output_acc = output.accessor<float, 2>();
        size_t pred_class = 0;
        float max_score = output_acc[0][0];
        for (size_t j = 1; j < MNISTDataset::sClasses; ++j) {
            if (output_acc[0][j] > max_score) {
                max_score = output_acc[0][j];
                pred_class = j;
            }
        }

        // Get true label
        auto target_acc = sample.target.accessor<float, 1>();
        size_t true_class = 0;
        for (size_t j = 0; j < MNISTDataset::sClasses; ++j) {
            if (target_acc[j] > 0.5f) {
                true_class = j;
                break;
            }
        }

        logger.info("Test Sample {}: Predicted={}, True={} {}", i, pred_class, true_class,
                    (pred_class == true_class) ? "✓" : "✗");
    }

    logger.info("=== MNIST Training Complete ===");
    Logger::shutdown();
    return 0;
}
