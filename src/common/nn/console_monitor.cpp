#include "loom/nn/console_monitor.h"

#include <iomanip>
#include <sstream>

#include "loom/logger.h"
#include "loom/nn/trainer.h"

namespace loom {
namespace nn {

ConsoleMonitor::ConsoleMonitor(const std::string& logger_name, bool show_progress)
    : mLoggerName(logger_name), mShowProgress(show_progress) {}

void ConsoleMonitor::onTrainStart(size_t total_epochs) {
    auto& logger = Logger::getInstance(mLoggerName);
    logger.info("=== Loom Training Started ===");
    logger.info("Training for {} epochs", total_epochs);
    logger.info("");
}

void ConsoleMonitor::onEpochStart(size_t epoch, size_t total_epochs) {
    auto& logger = Logger::getInstance(mLoggerName);
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    logger.info("Epoch {}/{}", epoch, total_epochs);
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    mEpochStart = std::chrono::steady_clock::now();
}

void ConsoleMonitor::onBatchComplete(size_t batch, size_t total_batches,
                                     [[maybe_unused]] float loss, [[maybe_unused]] float accuracy) {
    if (!mShowProgress)
        return;

    auto& logger = Logger::getInstance(mLoggerName);
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - mEpochStart).count();

    std::string progress = formatProgress(batch, total_batches, elapsed);
    logger.info("  {}", progress);
}

void ConsoleMonitor::onEpochEnd(const EpochMetrics& metrics) {
    auto& logger = Logger::getInstance(mLoggerName);
    logger.info("Epoch {} Summary (took {})", metrics.epoch,
                formatDuration(metrics.epoch_time_seconds));
    logger.info("┌─────────┬───────────┬──────────┐");
    logger.info("│ Metric  │   Train   │   Test   │");
    logger.info("├─────────┼───────────┼──────────┤");
    logger.info("│ Loss    │  {:.4f}   │  {:.4f}  │", metrics.train_loss, metrics.test_loss);
    logger.info("│ Acc     │  {:.2f}%  │  {:.2f}%  │", metrics.train_accuracy * 100,
                metrics.test_accuracy * 100);
    logger.info("│ Speed   │  {:.0f} samples/sec       │", metrics.samples_per_second);
    logger.info("└─────────┴───────────┴──────────┘");
}

void ConsoleMonitor::onTrainEnd(const TrainingHistory& history) {
    auto& logger = Logger::getInstance(mLoggerName);
    logger.info("");
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    logger.info("Training Complete!");
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if (history.size() > 0) {
        const auto& best = history.getBestEpoch(true);
        logger.info("Best Epoch: {} (Test Loss: {:.4f})", best.epoch, best.test_loss);
        logger.info("Average Test Accuracy: {:.2f}%", history.getAverageTestAccuracy() * 100);
    }
}

std::string ConsoleMonitor::formatProgress(size_t current, size_t total, double elapsed_sec) const {
    double percentage = (static_cast<double>(current) / total) * 100.0;

    // Estimate time remaining
    double avg_time_per_batch = elapsed_sec / current;
    double remaining_batches = total - current;
    double eta_sec = avg_time_per_batch * remaining_batches;

    std::ostringstream oss;
    oss << "Batch [" << current << "/" << total << "]" << std::fixed << std::setprecision(1) << " "
        << percentage << "%"
        << " │ ETA: " << formatDuration(eta_sec);

    return oss.str();
}

std::string ConsoleMonitor::formatDuration(double seconds) const {
    int total_sec = static_cast<int>(seconds);
    int hours = total_sec / 3600;
    int mins = (total_sec % 3600) / 60;
    int secs = total_sec % 60;

    std::ostringstream oss;
    if (hours > 0) {
        oss << std::setfill('0') << std::setw(2) << hours << ":" << std::setw(2) << mins << ":"
            << std::setw(2) << secs;
    } else {
        oss << std::setfill('0') << std::setw(2) << mins << ":" << std::setw(2) << secs;
    }

    return oss.str();
}

}  // namespace nn
}  // namespace loom
