#pragma once

#include <cstddef>

namespace loom {
namespace nn {

// Forward declarations
struct EpochMetrics;
class TrainingHistory;

/**
 * @brief Abstract interface for monitoring training progress.
 *
 * Allows pluggable output mechanisms (console, TUI, TensorBoard, etc.)
 * without modifying Trainer code. Enables future TUI integration.
 *
 * Example usage:
 *   auto monitor = std::make_shared<ConsoleMonitor>();
 *   Trainer trainer(model, optimizer, criterion, monitor);
 */
class TrainingMonitor {
   public:
    virtual ~TrainingMonitor() = default;

    /**
     * @brief Called when training starts.
     * @param total_epochs Total number of epochs to train
     */
    virtual void onTrainStart(size_t total_epochs) = 0;

    /**
     * @brief Called at the start of each epoch.
     * @param epoch Current epoch number (1-indexed)
     * @param total_epochs Total epochs
     */
    virtual void onEpochStart(size_t epoch, size_t total_epochs) = 0;

    /**
     * @brief Called after each training batch.
     * @param batch Current batch number (1-indexed)
     * @param total_batches Total batches in epoch
     * @param loss Current batch loss
     * @param accuracy Current batch accuracy
     */
    virtual void onBatchComplete(size_t batch, size_t total_batches, float loss,
                                 float accuracy) = 0;

    /**
     * @brief Called at the end of each epoch.
     * @param metrics Complete metrics for the epoch
     */
    virtual void onEpochEnd(const EpochMetrics& metrics) = 0;

    /**
     * @brief Called when training completes.
     * @param history Complete training history
     */
    virtual void onTrainEnd(const TrainingHistory& history) = 0;
};

}  // namespace nn
}  // namespace loom
