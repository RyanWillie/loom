#pragma once
#include <string>

namespace loom {
enum class DeviceType { CPU, CUDA, MPS };

class Device {
  public:
    Device(const DeviceType type, int index = 0);
    ~Device() = default;
    Device(const Device&) = default;
    Device& operator=(const Device&) = default;
    Device(Device&&) = default;
    Device& operator=(Device&&) = default;

    [[nodiscard]] int index() const;
    [[nodiscard]] DeviceType type() const;

    [[nodiscard]] bool isCPU() const;
    [[nodiscard]] bool isCUDA() const;
    [[nodiscard]] bool isMPS() const;

    bool operator==(const Device& other) const;
    bool operator!=(const Device& other) const;
    bool operator<(const Device& other) const;

    [[nodiscard]] std::string toString() const;

  private:
    DeviceType mType;
    int mIndex;
};

}  // namespace loom