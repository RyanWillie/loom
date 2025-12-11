#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "loom/device.h"
#include "loom/nn/parameter.h"
#include "loom/tensor/tensor.h"

namespace loom {
namespace nn {

/**
 * @brief Base class for all neural network modules.
 *
 * Module provides:
 * - Parameter registration and management
 * - Hierarchical composition (modules can contain submodules)
 * - Recursive parameter collection
 * - Device movement and gradient zeroing
 *
 * Design pattern:
 * - Subclasses register Parameters in constructor via registerParameter()
 * - Subclasses register sub-Modules via registerModule()
 * - Subclasses implement forward() for computation
 * - parameters() recursively collects all Parameters from tree
 */
class Module {
  public:
    Module() = default;
    virtual ~Module() = default;

    // Pure virtual: subclasses must implement forward pass
    virtual Tensor forward(const Tensor& input) = 0;

    // Convenience: call operator as shorthand for forward
    Tensor operator()(const Tensor& input) { return forward(input); }

    // ========================================================================
    // Parameter Registration
    // ========================================================================

    /**
     * @brief Register a parameter with a name.
     *
     * Stores the parameter in mParameters map and returns a shared_ptr to it.
     * The returned shared_ptr can be stored as a member variable for fast access.
     *
     * @param name Unique identifier for this parameter
     * @param param Parameter to register
     * @return shared_ptr to the registered parameter
     * @throws std::runtime_error if name already registered
     */
    std::shared_ptr<Parameter> registerParameter(const std::string& name, const Parameter& param);

    /**
     * @brief Register a submodule with a name.
     *
     * Stores the module in mSubmodules map and returns the shared_ptr.
     *
     * @param name Unique identifier for this submodule
     * @param module Submodule to register
     * @return shared_ptr to the registered module
     * @throws std::runtime_error if name already registered
     */
    std::shared_ptr<Module> registerModule(const std::string& name,
                                            std::shared_ptr<Module> module);

    // ========================================================================
    // Parameter Access
    // ========================================================================

    /**
     * @brief Get all parameters recursively.
     *
     * Collects parameters from this module and all submodules.
     * Order: own parameters first, then submodule parameters (DFS order).
     *
     * @return Vector of all parameters in the module tree
     */
    std::vector<std::shared_ptr<Parameter>> parameters() const;

    /**
     * @brief Get all parameters with their fully qualified names.
     *
     * Names are hierarchical: "layer1.weight", "layer1.bias", "layer2.weight", etc.
     *
     * @return Vector of (name, parameter) pairs
     */
    std::vector<std::pair<std::string, std::shared_ptr<Parameter>>> namedParameters() const;

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * @brief Move all parameters to a device.
     *
     * Recursively moves all parameters in this module and submodules.
     *
     * @param device Target device (CPU, CUDA, MPS)
     */
    void to(const Device& device);

    /**
     * @brief Zero all parameter gradients.
     *
     * Recursively zeros gradients for all parameters.
     */
    void zeroGrad();

    /**
     * @brief Set training mode.
     *
     * Affects layers like Dropout and BatchNorm (future).
     * Recursively sets mode for all submodules.
     *
     * @param mode true for training, false for evaluation
     */
    void train(bool mode = true);

    /**
     * @brief Set evaluation mode.
     *
     * Equivalent to train(false).
     */
    void eval() { train(false); }

    /**
     * @brief Check if in training mode.
     */
    bool training() const { return mTraining; }

  protected:
    // Named parameter storage
    // Key: parameter name (e.g., "weight", "bias")
    // Value: shared_ptr to Parameter
    std::map<std::string, std::shared_ptr<Parameter>> mParameters;

    // Named submodule storage
    // Key: submodule name (e.g., "layer1", "layer2")
    // Value: shared_ptr to Module
    std::map<std::string, std::shared_ptr<Module>> mSubmodules;

    // Training mode flag
    bool mTraining = true;

  private:
    // Helper for namedParameters() - performs DFS with prefix accumulation
    void namedParametersImpl(std::vector<std::pair<std::string, std::shared_ptr<Parameter>>>& result,
                             const std::string& prefix) const;
};

}  // namespace nn
}  // namespace loom
