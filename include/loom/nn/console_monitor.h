#pragma once

#include <chrono>
#include <string>

#include "loom/nn/training_monitor.h"

namespace loom {
namespace nn {

/**
 * @brief Default console-based training monitor.
 *
 * Provides TUI-inspired formatted output using the Logger system.
 * No external dependencies - works everywhere.
 *
 * Features:
 * - Clean visual separators for epochs
 * - Progress indicators with ETA
 * - Side-by-side train/test metrics in tables
 * - Throughput information (samples/sec)
 * - Summary statistics at the end
 *
 * Example output:
 *   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *   Epoch 1/5
 *   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *     Batch [100/938]  10.7% │ Loss: 0.4521 │ Acc: 85.3% │ ETA: 02:34
 *
 *   Epoch 1 Summary (took 00:02:45)
 *   ┌─────────┬───────────┬──────────┐
 *   │ Metric  │   Train   │   Test   │
 *   ├─────────┼───────────┼──────────┤
 *   │ Loss    │  0.3421   │  0.4123  │
 *   │ Acc     │  89.20%   │  87.40%  │
 *   │ Speed   │ 1247 samples/sec     │
 *   └─────────┴───────────┴──────────┘
 */
class ConsoleMonitor : public TrainingMonitor {
   public:
    /**
     * @brief Construct console monitor.
     * @param logger_name Name for logger instance (default: "Trainer")
     * @param show_progress Show progress indicators during training (default: true)
     */
    explicit ConsoleMonitor(const std::string& logger_name = "Trainer",
                           bool show_progress = true);

    // TrainingMonitor interface implementation
    void onTrainStart(size_t total_epochs) override;
    void onEpochStart(size_t epoch, size_t total_epochs) override;
    void onBatchComplete(size_t batch, size_t total_batches, float loss,
                        float accuracy) override;
    void onEpochEnd(const EpochMetrics& metrics) override;
    void onTrainEnd(const TrainingHistory& history) override;

   private:
    // Helper: format progress indicator with ETA
    std::string formatProgress(size_t current, size_t total, double elapsed_sec) const;

    // Helper: format duration as HH:MM:SS or MM:SS
    std::string formatDuration(double seconds) const;

    // Configuration
    std::string mLoggerName;
    bool mShowProgress;

    // Timing
    std::chrono::steady_clock::time_point mEpochStart;
};

}  // namespace nn
}  // namespace loom
