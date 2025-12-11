#include "loom/nn/sequential.h"

#include <stdexcept>

namespace loom {
namespace nn {

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    for (auto& module : modules) {
        add(module);
    }
}

Sequential& Sequential::add(std::shared_ptr<Module> module) {
    std::string name = std::to_string(mModules.size());
    registerModule(name, module);
    mModules.push_back(module);
    return *this;
}

Tensor Sequential::forward(const Tensor& input) {
    // Visualization:
    //   input -> module[0] -> output[0]
    //                      -> module[1] -> output[1]
    //                                   -> module[2] -> final output
    if (mModules.empty()) {
        throw std::runtime_error("Sequential is empty");
    }

    Tensor x = input;
    for (auto& module : mModules) {
        x = module->forward(x);
    }
    return x;
}

std::shared_ptr<Module> Sequential::operator[](size_t index) const {
    if (index >= mModules.size()) {
        throw std::out_of_range("Sequential index out of range");
    }
    return mModules[index];
}

}  // namespace nn
}  // namespace loom
