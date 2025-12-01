#pragma once

#include "common/dataloader/dataset.h"

namespace loom {

class MNISTDataset : public Dataset {
  public:
    MNISTDataset(const std::string& data_path, const std::string& label_path);

    [[nodiscard]] size_t size() const override;
    [[nodiscard]] Sample get(size_t index) const override;

    // MNIST dataset constants
    static constexpr size_t sImageSize = 28 * 28;
    static constexpr size_t sClasses = 10;

  private:
    Tensor mImages;
    Tensor mLabels;
    size_t mSize{0};

    void loadImages(const std::string& data_path);
    void loadLabels(const std::string& label_path);
    static uint32_t readBigEndianUint32(std::istream& stream);
};

}  // namespace loom