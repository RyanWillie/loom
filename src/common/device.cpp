#include "loom/device.h"

#include <stdexcept>
#include <string>

namespace loom {
Device::Device(const DeviceType type, int index) : mType(type), mIndex(index) {}

int Device::index() const {
    return mIndex;
}
DeviceType Device::type() const {
    return mType;
}
bool Device::isCPU() const {
    return mType == DeviceType::CPU;
}
bool Device::isCUDA() const {
    return mType == DeviceType::CUDA;
}
bool Device::isMPS() const {
    return mType == DeviceType::MPS;
}
bool Device::operator==(const Device& other) const {
    return mType == other.mType && mIndex == other.mIndex;
}
bool Device::operator!=(const Device& other) const {
    return !(*this == other);
}
bool Device::operator<(const Device& other) const {
    if (mType != other.mType) {
        return mType < other.mType;
    }
    return mIndex < other.mIndex;
}
std::string Device::toString() const {
    std::string result = "";
    switch (mType) {
        case DeviceType::CPU:
            result += "CPU";
            break;
        case DeviceType::CUDA:
            result += "CUDA";
            break;
        case DeviceType::MPS:
            result += "MPS";
            break;
        default:
            throw std::runtime_error("Unsupported device type");
            break;
    }
    result += "[" + std::to_string(mIndex) + "]";
    return result;
}
}  // namespace loom