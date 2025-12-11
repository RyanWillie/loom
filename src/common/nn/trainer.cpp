#include "loom/nn/trainer.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>

#include "loom/autograd/no_grad.h"
#include "loom/nn/console_monitor.h"

namespace loom {
namespace nn {

// ============================================================================
// TrainingHistory Implementation
// ============================================================================

void TrainingHistory::record(const EpochMetrics& metrics) {
    mHistory.push_back(metrics);
}

const std::vector<EpochMetrics>& TrainingHistory::getHistory() const {
    return mHistory;
}

size_t TrainingHistory::size() const {
    return mHistory.size();
}

const EpochMetrics& TrainingHistory::getBestEpoch(bool by_test_loss) const {
    if (mHistory.empty()) {
        throw std::runtime_error("TrainingHistory is empty");
    }
    if (by_test_loss) {
        return *std::min_element(
            mHistory.begin(), mHistory.end(),
            [](const EpochMetrics& a, const EpochMetrics& b) { return a.test_loss < b.test_loss; });
    } else {
        return *std::max_element(mHistory.begin(), mHistory.end(),
                                 [](const EpochMetrics& a, const EpochMetrics& b) {
                                     return a.test_accuracy < b.test_accuracy;
                                 });
    }
}

const EpochMetrics& TrainingHistory::getLatest() const {
    if (mHistory.empty()) {
        throw std::runtime_error("TrainingHistory is empty");
    }
    return mHistory.back();
}

void TrainingHistory::exportToJSON(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    file << "{\n";
    file << "  \"epochs\": [\n";

    for (size_t i = 0; i < mHistory.size(); ++i) {
        const auto& m = mHistory[i];
        file << "    {\n";
        file << "      \"epoch\": " << m.epoch << ",\n";
        file << "      \"train_loss\": " << m.train_loss << ",\n";
        file << "      \"train_accuracy\": " << m.train_accuracy << ",\n";
        file << "      \"test_loss\": " << m.test_loss << ",\n";
        file << "      \"test_accuracy\": " << m.test_accuracy << ",\n";
        file << "      \"epoch_time_seconds\": " << m.epoch_time_seconds << ",\n";
        file << "      \"samples_per_second\": " << m.samples_per_second << "\n";
        file << "    }";
        if (i < mHistory.size() - 1)
            file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";
    file.close();
}

void TrainingHistory::exportToCSV(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    // Write header
    file << "epoch,train_loss,train_accuracy,test_loss,test_accuracy,epoch_time_seconds,"
            "samples_per_second\n";

    // Write data rows
    for (const auto& m : mHistory) {
        file << m.epoch << "," << m.train_loss << "," << m.train_accuracy << "," << m.test_loss
             << "," << m.test_accuracy << "," << m.epoch_time_seconds << "," << m.samples_per_second
             << "\n";
    }

    file.close();
}

float TrainingHistory::getAverageTestAccuracy(size_t last_n) const {
    if (mHistory.empty())
        return 0.0f;

    size_t start_idx = 0;
    if (last_n > 0 && last_n < mHistory.size()) {
        start_idx = mHistory.size() - last_n;
    }

    float sum = 0.0f;
    size_t count = 0;
    for (size_t i = start_idx; i < mHistory.size(); ++i) {
        sum += mHistory[i].test_accuracy;
        count++;
    }

    return count > 0 ? sum / count : 0.0f;
}

// ============================================================================
// Trainer Implementation
// ============================================================================

Trainer::Trainer(std::shared_ptr<Module> model, std::shared_ptr<optim::Optimizer> optimizer,
                 std::shared_ptr<Loss> criterion, std::shared_ptr<TrainingMonitor> monitor)
    : mModel(model), mOptimizer(optimizer), mCriterion(criterion), mMonitor(monitor) {
    // Create default ConsoleMonitor if none provided
    if (!mMonitor) {
        mMonitor = std::make_shared<ConsoleMonitor>();
    }
}

Tensor Trainer::oneHotToIndices(const Tensor& one_hot) const {
    auto accessor = one_hot.accessor<float, 2>();
    size_t batch_size = one_hot.shape()[0];
    size_t num_classes = one_hot.shape()[1];

    Tensor result = Tensor::zeros({batch_size}, DType::INT64, one_hot.device());
    auto result_acc = result.accessor<int64_t, 1>();

    for (size_t i = 0; i < batch_size; ++i) {
        size_t max_idx = 0;
        float max_val = accessor[i][0];
        for (size_t j = 1; j < num_classes; ++j) {
            if (accessor[i][j] > max_val) {
                max_val = accessor[i][j];
                max_idx = j;
            }
        }
        result_acc[i] = static_cast<int64_t>(max_idx);
    }

    return result;
}

float Trainer::computeAccuracy(const Tensor& predictions, const Tensor& targets) const {
    auto pred_acc = predictions.accessor<float, 2>();
    auto target_acc = targets.accessor<int64_t, 1>();

    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];
    size_t correct = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        size_t pred_class = 0;
        float max_score = pred_acc[i][0];
        for (size_t j = 1; j < num_classes; ++j) {
            if (pred_acc[i][j] > max_score) {
                max_score = pred_acc[i][j];
                pred_class = j;
            }
        }

        if (static_cast<int64_t>(pred_class) == target_acc[i]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / batch_size;
}

std::pair<float, float> Trainer::trainEpoch(DataLoader& train_loader, size_t log_interval) {
    [[maybe_unused]] auto epoch_start = std::chrono::steady_clock::now();

    float total_loss = 0.0f;
    float total_accuracy = 0.0f;
    size_t num_batches = train_loader.numBatches();
    size_t total_samples = 0;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto batch = train_loader.getBatch(batch_idx);
        Tensor targets = oneHotToIndices(batch.target);

        // Training step
        mOptimizer->zeroGrad();
        Tensor logits = mModel->forward(batch.input);
        Tensor loss = mCriterion->forward(logits, targets);
        loss.backward();
        mOptimizer->step();

        // Track metrics
        float batch_loss = loss.item();
        float batch_acc = computeAccuracy(logits, targets);
        total_loss += batch_loss;
        total_accuracy += batch_acc;
        total_samples += batch.input.shape()[0];

        // Monitor callback
        if ((batch_idx + 1) % log_interval == 0 && mMonitor) {
            float avg_loss = total_loss / (batch_idx + 1);
            float avg_acc = total_accuracy / (batch_idx + 1);
            mMonitor->onBatchComplete(batch_idx + 1, num_batches, avg_loss, avg_acc);
        }
    }

    return {total_loss / num_batches, total_accuracy / num_batches};
}

std::pair<float, float> Trainer::evaluate(DataLoader& data_loader) const {
    autograd::NoGrad no_grad;  // Disable gradient tracking

    float total_loss = 0.0f;
    float total_accuracy = 0.0f;
    size_t num_batches = data_loader.numBatches();

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto batch = data_loader.getBatch(batch_idx);
        Tensor targets = oneHotToIndices(batch.target);

        // Forward pass only
        Tensor logits = mModel->forward(batch.input);
        Tensor loss = mCriterion->forward(logits, targets);

        total_loss += loss.item();
        total_accuracy += computeAccuracy(logits, targets);
    }

    return {total_loss / num_batches, total_accuracy / num_batches};
}

const TrainingHistory& Trainer::train(DataLoader& train_loader, DataLoader& test_loader,
                                      size_t num_epochs, size_t log_interval) {
    // Clear history for fresh start
    mHistory = TrainingHistory();

    // Notify training start
    if (mMonitor) {
        mMonitor->onTrainStart(num_epochs);
    }

    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        // Notify epoch start
        if (mMonitor) {
            mMonitor->onEpochStart(epoch, num_epochs);
        }

        // Start timing
        auto epoch_start = std::chrono::steady_clock::now();

        // Training phase
        auto [train_loss, train_acc] = trainEpoch(train_loader, log_interval);

        // Evaluation phase
        auto [test_loss, test_acc] = evaluate(test_loader);

        // Calculate timing metrics
        auto epoch_end = std::chrono::steady_clock::now();
        double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();

        // Calculate throughput (samples per second)
        size_t total_samples =
            train_loader.numBatches() * train_loader.getBatch(0).input.shape()[0];
        double samples_per_sec = total_samples / epoch_time;

        // Record metrics
        EpochMetrics metrics;
        metrics.epoch = epoch;
        metrics.train_loss = train_loss;
        metrics.train_accuracy = train_acc;
        metrics.test_loss = test_loss;
        metrics.test_accuracy = test_acc;
        metrics.epoch_time_seconds = epoch_time;
        metrics.samples_per_second = samples_per_sec;

        mHistory.record(metrics);

        // Notify epoch end
        if (mMonitor) {
            mMonitor->onEpochEnd(metrics);
        }

        // Reset data loader for next epoch
        train_loader.reset();
    }

    // Notify training complete
    if (mMonitor) {
        mMonitor->onTrainEnd(mHistory);
    }

    return mHistory;
}

const TrainingHistory& Trainer::history() const {
    return mHistory;
}

}  // namespace nn
}  // namespace loom
