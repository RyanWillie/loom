#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "loom/nn/module.h"
#include "loom/tensor/tensor.h"

namespace loom {
namespace nn {

/**
 * @brief Sequential container for chaining modules.
 *
 * Sequential applies a sequence of modules in order, passing the output
 * of each module as input to the next. This is the most common way to
 * build feed-forward neural networks.
 *
 * Modules are automatically registered with numeric names ("0", "1", "2", ...)
 * so parameters() recursively collects all parameters from all submodules.
 *
 * Example usage:
 *   auto model = Sequential({
 *       std::make_shared<Linear>(784, 128),
 *       std::make_shared<Linear>(128, 10)
 *   });
 *
 *   Tensor x = Tensor::randn({32, 784});
 *   Tensor y = model->forward(x);  // shape: [32, 10]
 *
 * Builder pattern:
 *   Sequential model;
 *   model.add(std::make_shared<Linear>(784, 128));
 *   model.add(std::make_shared<Linear>(128, 10));
 */
class Sequential : public Module {
  public:
    /**
     * @brief Construct an empty Sequential container.
     */
    Sequential() = default;

    /**
     * @brief Construct Sequential from initializer list.
     *
     * @param modules List of modules to add in order
     *
     * Example:
     *   Sequential({layer1, layer2, layer3});
     */
    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);

    /**
     * @brief Add a module to the end of the sequence.
     *
     * Modules are automatically named "0", "1", "2", etc. based on order.
     *
     * @param module Module to add
     * @return Reference to this Sequential for chaining
     *
     * Example:
     *   model.add(layer1).add(layer2).add(layer3);
     */
    Sequential& add(std::shared_ptr<Module> module);

    /**
     * @brief Forward pass: chain module outputs.
     *
     * Applies modules in order: output = module_n(...module_1(input))
     *
     * @param input Input tensor
     * @return Output tensor after passing through all modules
     * @throws std::runtime_error if no modules have been added
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Get number of modules in the sequence.
     */
    size_t size() const { return mModules.size(); }

    /**
     * @brief Check if sequence is empty.
     */
    bool empty() const { return mModules.empty(); }

    /**
     * @brief Get module at index (0-indexed).
     *
     * @param index Index of module to retrieve
     * @return Shared pointer to module
     * @throws std::out_of_range if index >= size()
     */
    std::shared_ptr<Module> operator[](size_t index) const;

  private:
    // Vector of modules in order
    std::vector<std::shared_ptr<Module>> mModules;
};

}  // namespace nn
}  // namespace loom
