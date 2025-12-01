#pragma once
#include "common/dataloader/dataset.h"
#include "common/tensor/tensor.h"

namespace loom {

struct Batch {
    Tensor input;   // Shape: [batch_size, ...]
    Tensor target;  // Shape: [batch_size, ...]
};

// Batching, shuffling, etc.
class DataLoader {
  public:
    DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle = false);
    ~DataLoader() = default;

    [[nodiscard]] size_t numBatches() const;  // Number of batches
    [[nodiscard]] Batch getBatch(size_t index) const;
    void reset();  // Reshuffle for next epoch

  private:
    const Dataset& mDataset;
    size_t mBatchSize;
    bool mShuffle;
    std::vector<size_t> mIndices;
};

}  // namespace loom
