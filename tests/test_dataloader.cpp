#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "loom/dataloader/dataloader.h"
#include "loom/dataloader/dataset.h"
#include "loom/device.h"
#include "loom/dtypes.h"
#include "loom/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace loom;

// ============================================================================
// Test Dataset Implementation
// ============================================================================

/**
 * Simple in-memory dataset for testing.
 * Each sample has:
 *   - input: 1D tensor of size input_size with value = index (for verification)
 *   - target: 1D tensor of size target_size with value = index * 2
 */
class TestDataset : public Dataset {
  public:
    TestDataset(size_t num_samples, size_t input_size, size_t target_size)
        : mNumSamples(num_samples),
          mInputSize(input_size),
          mTargetSize(target_size),
          mInputs(Tensor::zeros({num_samples, input_size}, DType::FLOAT32)),
          mTargets(Tensor::zeros({num_samples, target_size}, DType::FLOAT32)) {
        if (input_size > 0 && target_size > 0) {
            auto input_acc = mInputs.accessor<float, 2>();
            auto target_acc = mTargets.accessor<float, 2>();

            for (size_t i = 0; i < num_samples; ++i) {
                for (size_t j = 0; j < input_size; ++j) {
                    input_acc[i][j] = static_cast<float>(i);  // Use index as value
                }
                for (size_t j = 0; j < target_size; ++j) {
                    target_acc[i][j] = static_cast<float>(i * 2);  // Use 2x index as target
                }
            }
        }
    }

    [[nodiscard]] size_t size() const override { return mNumSamples; }

    [[nodiscard]] Sample get(size_t index) const override {
        return {mInputs.slice(0, index, index + 1).squeeze(),
                mTargets.slice(0, index, index + 1).squeeze()};
    }

  private:
    size_t mNumSamples;
    size_t mInputSize;
    size_t mTargetSize;
    Tensor mInputs;
    Tensor mTargets;
};

// ============================================================================
// Dataset Tests
// ============================================================================

class DatasetTest : public ::testing::Test {
  protected:
    Device cpu_device{DeviceType::CPU};
};

TEST_F(DatasetTest, TestDatasetSize) {
    TestDataset dataset(100, 10, 5);
    EXPECT_EQ(dataset.size(), 100);
}

TEST_F(DatasetTest, TestDatasetGetSample) {
    TestDataset dataset(10, 8, 4);

    Sample sample = dataset.get(3);

    // Check input shape
    EXPECT_EQ(sample.input.ndim(), 1);
    EXPECT_EQ(sample.input.size(0), 8);

    // Check target shape
    EXPECT_EQ(sample.target.ndim(), 1);
    EXPECT_EQ(sample.target.size(0), 4);

    // Check values (index 3 should have input=3.0, target=6.0)
    auto input_acc = sample.input.accessor<float, 1>();
    auto target_acc = sample.target.accessor<float, 1>();

    EXPECT_FLOAT_EQ(input_acc[0], 3.0f);
    EXPECT_FLOAT_EQ(target_acc[0], 6.0f);
}

TEST_F(DatasetTest, TestDatasetFirstAndLastSample) {
    TestDataset dataset(50, 5, 3);

    // First sample (index 0)
    Sample first = dataset.get(0);
    auto first_input = first.input.accessor<float, 1>();
    EXPECT_FLOAT_EQ(first_input[0], 0.0f);

    // Last sample (index 49)
    Sample last = dataset.get(49);
    auto last_input = last.input.accessor<float, 1>();
    EXPECT_FLOAT_EQ(last_input[0], 49.0f);
}

TEST_F(DatasetTest, TestDatasetSingleSample) {
    TestDataset dataset(1, 10, 5);
    EXPECT_EQ(dataset.size(), 1);

    Sample sample = dataset.get(0);
    EXPECT_EQ(sample.input.size(0), 10);
    EXPECT_EQ(sample.target.size(0), 5);
}

TEST_F(DatasetTest, TestDatasetMinimalSize) {
    // Dataset with minimal sizes
    TestDataset dataset(2, 1, 1);
    EXPECT_EQ(dataset.size(), 2);

    Sample sample = dataset.get(0);
    EXPECT_EQ(sample.input.numel(), 1);
    EXPECT_EQ(sample.target.numel(), 1);
}

// ============================================================================
// DataLoader Tests
// ============================================================================

class DataLoaderTest : public ::testing::Test {
  protected:
    Device cpu_device{DeviceType::CPU};
};

TEST_F(DataLoaderTest, ConstructorNoShuffle) {
    TestDataset dataset(100, 10, 5);
    DataLoader loader(dataset, 10, false);

    EXPECT_EQ(loader.numBatches(), 10);  // 100 samples / 10 batch_size = 10 batches
}

TEST_F(DataLoaderTest, ConstructorWithShuffle) {
    TestDataset dataset(100, 10, 5);
    DataLoader loader(dataset, 10, true);

    EXPECT_EQ(loader.numBatches(), 10);
}

TEST_F(DataLoaderTest, NumBatchesExactDivision) {
    TestDataset dataset(64, 8, 4);
    DataLoader loader(dataset, 8, false);

    EXPECT_EQ(loader.numBatches(), 8);  // 64 / 8 = 8 exact
}

TEST_F(DataLoaderTest, NumBatchesDropsIncomplete) {
    TestDataset dataset(100, 10, 5);
    DataLoader loader(dataset, 32, false);

    // 100 / 32 = 3.125, so we should have 3 complete batches
    EXPECT_EQ(loader.numBatches(), 3);
}

TEST_F(DataLoaderTest, NumBatchesBatchSizeOne) {
    TestDataset dataset(10, 5, 3);
    DataLoader loader(dataset, 1, false);

    EXPECT_EQ(loader.numBatches(), 10);
}

TEST_F(DataLoaderTest, NumBatchesBatchSizeEqualsDatasetSize) {
    TestDataset dataset(50, 10, 5);
    DataLoader loader(dataset, 50, false);

    EXPECT_EQ(loader.numBatches(), 1);  // One big batch
}

TEST_F(DataLoaderTest, NumBatchesBatchSizeLargerThanDataset) {
    TestDataset dataset(10, 5, 3);
    DataLoader loader(dataset, 100, false);

    EXPECT_EQ(loader.numBatches(), 0);  // No complete batches
}

TEST_F(DataLoaderTest, GetBatchShape) {
    TestDataset dataset(100, 10, 5);
    DataLoader loader(dataset, 16, false);

    Batch batch = loader.getBatch(0);

    // Check batch input shape: [batch_size, input_size]
    EXPECT_EQ(batch.input.ndim(), 2);
    EXPECT_EQ(batch.input.size(0), 16);
    EXPECT_EQ(batch.input.size(1), 10);

    // Check batch target shape: [batch_size, target_size]
    EXPECT_EQ(batch.target.ndim(), 2);
    EXPECT_EQ(batch.target.size(0), 16);
    EXPECT_EQ(batch.target.size(1), 5);
}

TEST_F(DataLoaderTest, GetBatchValuesNoShuffle) {
    TestDataset dataset(20, 4, 2);
    DataLoader loader(dataset, 5, false);

    Batch batch = loader.getBatch(0);

    auto input_acc = batch.input.accessor<float, 2>();
    auto target_acc = batch.target.accessor<float, 2>();

    // Without shuffle, first batch should have samples 0-4
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(input_acc[i][0], static_cast<float>(i));
        EXPECT_FLOAT_EQ(target_acc[i][0], static_cast<float>(i * 2));
    }
}

TEST_F(DataLoaderTest, GetBatchSecondBatch) {
    TestDataset dataset(20, 4, 2);
    DataLoader loader(dataset, 5, false);

    Batch batch = loader.getBatch(1);  // Second batch

    auto input_acc = batch.input.accessor<float, 2>();

    // Without shuffle, second batch should have samples 5-9
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(input_acc[i][0], static_cast<float>(i + 5));
    }
}

TEST_F(DataLoaderTest, GetBatchLastBatch) {
    TestDataset dataset(32, 4, 2);
    DataLoader loader(dataset, 10, false);

    // 32 / 10 = 3 complete batches, last batch is batch 2
    Batch batch = loader.getBatch(2);

    auto input_acc = batch.input.accessor<float, 2>();

    // Last complete batch should have samples 20-29
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(input_acc[i][0], static_cast<float>(i + 20));
    }
}

TEST_F(DataLoaderTest, GetBatchDTypeIsFloat32) {
    TestDataset dataset(50, 10, 5);
    DataLoader loader(dataset, 8, false);

    Batch batch = loader.getBatch(0);

    EXPECT_EQ(batch.input.dtype(), DType::FLOAT32);
    EXPECT_EQ(batch.target.dtype(), DType::FLOAT32);
}

TEST_F(DataLoaderTest, ShuffleChangesOrder) {
    TestDataset dataset(100, 4, 2);

    // Get first batch without shuffle
    DataLoader loader_no_shuffle(dataset, 10, false);
    Batch batch_no_shuffle = loader_no_shuffle.getBatch(0);

    // Get first batch with shuffle
    DataLoader loader_shuffle(dataset, 10, true);
    Batch batch_shuffle = loader_shuffle.getBatch(0);

    // Compare values - with high probability, shuffled batch will be different
    // Note: There's a tiny chance they could be the same, but it's astronomically unlikely
    auto no_shuffle_acc = batch_no_shuffle.input.accessor<float, 2>();
    auto shuffle_acc = batch_shuffle.input.accessor<float, 2>();

    bool all_same = true;
    for (size_t i = 0; i < 10 && all_same; ++i) {
        if (std::abs(no_shuffle_acc[i][0] - shuffle_acc[i][0]) > 1e-6f) {
            all_same = false;
        }
    }

    // With 100 samples and batch size 10, the probability of getting the exact same
    // first 10 in the same order is vanishingly small
    EXPECT_FALSE(all_same) << "Shuffled batch should differ from unshuffled batch";
}

TEST_F(DataLoaderTest, ShufflePreservesAllIndices) {
    TestDataset dataset(20, 4, 2);
    DataLoader loader(dataset, 20, true);  // All samples in one batch

    Batch batch = loader.getBatch(0);
    auto input_acc = batch.input.accessor<float, 2>();

    // Collect all values from the batch
    std::set<float> seen_values;
    for (size_t i = 0; i < 20; ++i) {
        seen_values.insert(input_acc[i][0]);
    }

    // All indices 0-19 should be present
    EXPECT_EQ(seen_values.size(), 20);
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_TRUE(seen_values.count(static_cast<float>(i)) > 0)
            << "Missing index " << i << " after shuffle";
    }
}

TEST_F(DataLoaderTest, ResetReshuffles) {
    TestDataset dataset(50, 4, 2);
    DataLoader loader(dataset, 50, true);  // All in one batch

    // Get batch before reset
    Batch batch1 = loader.getBatch(0);
    std::vector<float> order1;
    {
        auto acc = batch1.input.accessor<float, 2>();
        for (size_t i = 0; i < 50; ++i) {
            order1.push_back(acc[i][0]);
        }
    }

    // Reset (reshuffle)
    loader.reset();

    // Get batch after reset
    Batch batch2 = loader.getBatch(0);
    std::vector<float> order2;
    {
        auto acc = batch2.input.accessor<float, 2>();
        for (size_t i = 0; i < 50; ++i) {
            order2.push_back(acc[i][0]);
        }
    }

    // Orders should (almost certainly) be different
    EXPECT_NE(order1, order2) << "Reset should change the order";
}

TEST_F(DataLoaderTest, ResetWithoutShuffleDoesNothing) {
    TestDataset dataset(20, 4, 2);
    DataLoader loader(dataset, 5, false);  // No shuffle

    // Get batches before reset
    Batch batch1 = loader.getBatch(0);
    auto acc1 = batch1.input.accessor<float, 2>();
    std::vector<float> before;
    for (size_t i = 0; i < 5; ++i) {
        before.push_back(acc1[i][0]);
    }

    // Reset
    loader.reset();

    // Get batches after reset
    Batch batch2 = loader.getBatch(0);
    auto acc2 = batch2.input.accessor<float, 2>();
    std::vector<float> after;
    for (size_t i = 0; i < 5; ++i) {
        after.push_back(acc2[i][0]);
    }

    // Order should be the same (no shuffle)
    EXPECT_EQ(before, after);
}

TEST_F(DataLoaderTest, MultipleBatchesNoOverlap) {
    TestDataset dataset(30, 4, 2);
    DataLoader loader(dataset, 10, false);

    std::set<float> seen_indices;

    // Get all 3 batches
    for (size_t b = 0; b < loader.numBatches(); ++b) {
        Batch batch = loader.getBatch(b);
        auto acc = batch.input.accessor<float, 2>();

        for (size_t i = 0; i < 10; ++i) {
            float idx = acc[i][0];
            EXPECT_EQ(seen_indices.count(idx), 0) << "Index " << idx << " seen in multiple batches";
            seen_indices.insert(idx);
        }
    }

    EXPECT_EQ(seen_indices.size(), 30);
}

TEST_F(DataLoaderTest, SmallBatchSize) {
    TestDataset dataset(10, 5, 3);
    DataLoader loader(dataset, 2, false);

    EXPECT_EQ(loader.numBatches(), 5);

    for (size_t b = 0; b < loader.numBatches(); ++b) {
        Batch batch = loader.getBatch(b);
        EXPECT_EQ(batch.input.size(0), 2);
        EXPECT_EQ(batch.target.size(0), 2);
    }
}

TEST_F(DataLoaderTest, LargeBatchSize) {
    TestDataset dataset(1000, 10, 5);
    DataLoader loader(dataset, 256, false);

    EXPECT_EQ(loader.numBatches(), 3);  // 1000 / 256 = 3.90625

    Batch batch = loader.getBatch(0);
    EXPECT_EQ(batch.input.size(0), 256);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DataLoaderTest, SingleSampleDataset) {
    TestDataset dataset(1, 10, 5);
    DataLoader loader(dataset, 1, false);

    EXPECT_EQ(loader.numBatches(), 1);

    Batch batch = loader.getBatch(0);
    EXPECT_EQ(batch.input.size(0), 1);
    EXPECT_EQ(batch.input.size(1), 10);
}

TEST_F(DataLoaderTest, BatchSizeOneFullIteration) {
    TestDataset dataset(5, 3, 2);
    DataLoader loader(dataset, 1, false);

    EXPECT_EQ(loader.numBatches(), 5);

    for (size_t b = 0; b < 5; ++b) {
        Batch batch = loader.getBatch(b);
        auto acc = batch.input.accessor<float, 2>();
        EXPECT_FLOAT_EQ(acc[0][0], static_cast<float>(b));
    }
}

TEST_F(DataLoaderTest, IterateAllBatchesTwice) {
    TestDataset dataset(24, 6, 4);
    DataLoader loader(dataset, 8, false);

    EXPECT_EQ(loader.numBatches(), 3);

    // First epoch
    for (size_t b = 0; b < loader.numBatches(); ++b) {
        Batch batch = loader.getBatch(b);
        EXPECT_EQ(batch.input.size(0), 8);
    }

    // Second epoch (same order since no shuffle)
    for (size_t b = 0; b < loader.numBatches(); ++b) {
        Batch batch = loader.getBatch(b);
        EXPECT_EQ(batch.input.size(0), 8);
    }
}

TEST_F(DataLoaderTest, LargeDataset) {
    TestDataset dataset(10000, 20, 10);
    DataLoader loader(dataset, 128, false);

    EXPECT_EQ(loader.numBatches(), 78);  // 10000 / 128 = 78.125

    // Just verify first and last batch work
    Batch first = loader.getBatch(0);
    EXPECT_EQ(first.input.size(0), 128);

    Batch last = loader.getBatch(77);
    EXPECT_EQ(last.input.size(0), 128);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(DataLoaderTest, SimulateTrainingLoop) {
    TestDataset dataset(100, 10, 1);
    DataLoader loader(dataset, 16, true);

    const size_t num_epochs = 3;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::set<float> epoch_samples;

        for (size_t batch_idx = 0; batch_idx < loader.numBatches(); ++batch_idx) {
            Batch batch = loader.getBatch(batch_idx);

            // Verify batch shapes
            EXPECT_EQ(batch.input.size(0), 16);
            EXPECT_EQ(batch.input.size(1), 10);
            EXPECT_EQ(batch.target.size(0), 16);
            EXPECT_EQ(batch.target.size(1), 1);

            // Track which samples we've seen
            auto acc = batch.input.accessor<float, 2>();
            for (size_t i = 0; i < 16; ++i) {
                epoch_samples.insert(acc[i][0]);
            }
        }

        // With 100 samples and batch_size 16, we get 6 complete batches = 96 samples
        EXPECT_EQ(epoch_samples.size(), 96);

        // Reset for next epoch
        loader.reset();
    }
}

TEST_F(DataLoaderTest, ConsistentSampleIndicesWithShuffleFalse) {
    TestDataset dataset(50, 8, 4);
    DataLoader loader(dataset, 10, false);

    // First pass
    std::vector<float> first_pass;
    for (size_t b = 0; b < loader.numBatches(); ++b) {
        Batch batch = loader.getBatch(b);
        auto acc = batch.input.accessor<float, 2>();
        for (size_t i = 0; i < 10; ++i) {
            first_pass.push_back(acc[i][0]);
        }
    }

    // Second pass (should be identical)
    std::vector<float> second_pass;
    for (size_t b = 0; b < loader.numBatches(); ++b) {
        Batch batch = loader.getBatch(b);
        auto acc = batch.input.accessor<float, 2>();
        for (size_t i = 0; i < 10; ++i) {
            second_pass.push_back(acc[i][0]);
        }
    }

    EXPECT_EQ(first_pass, second_pass);
}
