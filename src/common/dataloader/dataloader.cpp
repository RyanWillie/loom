#include "loom/dataloader/dataloader.h"

#include <algorithm>
#include <cstring>
#include <random>

namespace loom {

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle)
    : mDataset(dataset), mBatchSize(batch_size), mShuffle(shuffle) {
    // Initialize indices [0, 1, 2, ..., N-1]
    mIndices.resize(dataset.size());
    for (size_t i = 0; i < mIndices.size(); ++i) {
        mIndices[i] = i;
    }

    // Shuffle on construction if requested
    if (mShuffle) {
        reset();
    }
}

size_t DataLoader::numBatches() const {
    // Number of complete batches (drops incomplete last batch)
    return mDataset.size() / mBatchSize;
}

void DataLoader::reset() {
    if (mShuffle) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(mIndices.begin(), mIndices.end(), gen);
    }
}

Batch DataLoader::getBatch(size_t batch_index) const {
    size_t start = batch_index * mBatchSize;
    size_t end = std::min(start + mBatchSize, mDataset.size());
    size_t actual_size = end - start;

    // Get first sample to determine shapes
    Sample first_sample = mDataset.get(mIndices[start]);
    size_t input_size = first_sample.input.numel();
    size_t target_size = first_sample.target.numel();

    // Create batch tensors
    Tensor batch_input = Tensor::zeros({actual_size, input_size}, DType::FLOAT32);
    Tensor batch_target = Tensor::zeros({actual_size, target_size}, DType::FLOAT32);

    // Fill using accessors
    auto input_acc = batch_input.accessor<float, 2>();
    auto target_acc = batch_target.accessor<float, 2>();

    for (size_t i = 0; i < actual_size; ++i) {
        size_t sample_idx = mIndices[start + i];
        Sample sample = mDataset.get(sample_idx);

        // Copy entire row at once using memcpy (samples are contiguous)
        auto sample_input = sample.input.accessor<float, 1>();
        auto sample_target = sample.target.accessor<float, 1>();

        std::memcpy(&input_acc[i][0], &sample_input[0], input_size * sizeof(DType::FLOAT32));
        std::memcpy(&target_acc[i][0], &sample_target[0], target_size * sizeof(DType::FLOAT32));
    }

    return {batch_input, batch_target};
}

}  // namespace loom
