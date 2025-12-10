#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "loom/dataloader/dataloader.h"
#include "loom/nn/loss.h"
#include "loom/nn/module.h"
#include "loom/nn/training_monitor.h"
#include "loom/optim/optimizer.h"

namespace loom {
namespace nn {

/**
 * @brief Metrics for a single training epoch.
 *
 * Captures both performance metrics (loss, accuracy) and timing information
 * for analysis and visualization.
 */
struct EpochMetrics {
    size_t epoch = 0;

    // Training metrics
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;

    // Validation/Test metrics
    float test_loss = 0.0f;
    float test_accuracy = 0.0f;

    // Timing information
    double epoch_time_seconds = 0.0;
    double samples_per_second = 0.0;
};

/**
 * @brief Training history tracker for storing and analyzing epoch metrics.
 *
 * Provides query methods and export capabilities for post-training analysis.
 */
class TrainingHistory {
   public:
    /**
     * @brief Record metrics for an epoch.
     * @param metrics Epoch metrics to record
     */
    void record(const EpochMetrics& metrics);

    /**
     * @brief Get the complete training history.
     * @return Vector of all recorded epoch metrics
     */
    const std::vector<EpochMetrics>& getHistory() const;

    /**
     * @brief Get the number of recorded epochs.
     */
    size_t size() const;

    /**
     * @brief Get the epoch with the best performance.
     * @param by_test_loss If true, use test loss (lower is better); otherwise use test accuracy (higher is better)
     * @return Reference to the best epoch's metrics
     * @throws std::runtime_error if history is empty
     */
    const EpochMetrics& getBestEpoch(bool by_test_loss = true) const;

    /**
     * @brief Get the most recent epoch's metrics.
     * @return Reference to the latest epoch's metrics
     * @throws std::runtime_error if history is empty
     */
    const EpochMetrics& getLatest() const;

    /**
     * @brief Export training history to JSON format.
     * @param filepath Path to output JSON file
     */
    void exportToJSON(const std::string& filepath) const;

    /**
     * @brief Export training history to CSV format.
     * @param filepath Path to output CSV file
     */
    void exportToCSV(const std::string& filepath) const;

    /**
     * @brief Calculate average test accuracy.
     * @param last_n Number of recent epochs to average (0 = all epochs)
     * @return Average test accuracy
     */
    float getAverageTestAccuracy(size_t last_n = 0) const;

   private:
    std::vector<EpochMetrics> mHistory;
};

/**
 * @brief Main training orchestrator for neural network models.
 *
 * Encapsulates the training loop logic, eliminating repetitive boilerplate code.
 * Supports pluggable monitoring through the TrainingMonitor interface.
 *
 * Example usage:
 *   auto model = std::make_shared<nn::Sequential>(...);
 *   auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
 *   auto criterion = std::make_shared<nn::CrossEntropyLoss>();
 *
 *   // Create trainer (uses default ConsoleMonitor)
 *   auto trainer = std::make_shared<nn::Trainer>(model, optimizer, criterion);
 *
 *   // Train for 5 epochs
 *   const auto& history = trainer->train(train_loader, test_loader, 5);
 *
 *   // Access metrics
 *   const auto& best = history.getBestEpoch();
 */
class Trainer {
   public:
    /**
     * @brief Construct a Trainer.
     * @param model Neural network model
     * @param optimizer Optimizer for parameter updates
     * @param criterion Loss function
     * @param monitor Optional training monitor (defaults to ConsoleMonitor if nullptr)
     */
    Trainer(std::shared_ptr<Module> model, std::shared_ptr<optim::Optimizer> optimizer,
            std::shared_ptr<Loss> criterion,
            std::shared_ptr<TrainingMonitor> monitor = nullptr);

    /**
     * @brief Train for multiple epochs.
     * @param train_loader Training data loader
     * @param test_loader Test/validation data loader
     * @param num_epochs Number of epochs to train
     * @param log_interval Log progress every N batches (default: 100)
     * @return Reference to training history
     */
    const TrainingHistory& train(DataLoader& train_loader, DataLoader& test_loader,
                                  size_t num_epochs, size_t log_interval = 100);

    /**
     * @brief Train for a single epoch.
     * @param train_loader Training data loader
     * @param log_interval Log progress every N batches
     * @return (average_loss, average_accuracy)
     */
    std::pair<float, float> trainEpoch(DataLoader& train_loader, size_t log_interval = 100);

    /**
     * @brief Evaluate model on a dataset.
     * @param data_loader Data loader for evaluation
     * @return (average_loss, average_accuracy)
     */
    std::pair<float, float> evaluate(DataLoader& data_loader) const;

    /**
     * @brief Access the training history.
     * @return Reference to training history
     */
    const TrainingHistory& history() const;

   private:
    // Helper: convert one-hot encoded targets to class indices
    Tensor oneHotToIndices(const Tensor& one_hot) const;

    // Helper: compute classification accuracy
    float computeAccuracy(const Tensor& predictions, const Tensor& targets) const;

    // Components
    std::shared_ptr<Module> mModel;
    std::shared_ptr<optim::Optimizer> mOptimizer;
    std::shared_ptr<Loss> mCriterion;
    std::shared_ptr<TrainingMonitor> mMonitor;

    // History tracking
    TrainingHistory mHistory;

    // Timing
    std::chrono::steady_clock::time_point mEpochStart;
};

}  // namespace nn
}  // namespace loom
