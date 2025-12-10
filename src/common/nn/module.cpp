#include "loom/nn/module.h"

#include <stdexcept>

namespace loom {
namespace nn {

// ============================================================================
// Parameter Registration
// ============================================================================

std::shared_ptr<Parameter> Module::registerParameter(const std::string& name,
                                                     const Parameter& param) {
    // Check for duplicate names
    if (mParameters.find(name) != mParameters.end()) {
        throw std::runtime_error("Parameter '" + name + "' already registered");
    }

    // Create shared_ptr and store in map
    auto param_ptr = std::make_shared<Parameter>(param);
    mParameters[name] = param_ptr;

    return param_ptr;
}

std::shared_ptr<Module> Module::registerModule(const std::string& name,
                                               std::shared_ptr<Module> module) {
    // Check for duplicate names
    if (mSubmodules.find(name) != mSubmodules.end()) {
        throw std::runtime_error("Submodule '" + name + "' already registered");
    }

    // Store in map
    mSubmodules[name] = module;

    return module;
}

// ============================================================================
// Parameter Access
// ============================================================================

std::vector<std::shared_ptr<Parameter>> Module::parameters() const {
    // Recursive parameter collection using Depth-First Search
    // Collects own parameters first, then recursively collects from submodules
    std::vector<std::shared_ptr<Parameter>> result;
    for (auto& [name, param] : mParameters) {
        result.push_back(param);
    }
    for (auto& [name, submodule] : mSubmodules) {
        auto subparams = submodule->parameters();
        result.insert(result.end(), subparams.begin(), subparams.end());
    }
    return result;
}

std::vector<std::pair<std::string, std::shared_ptr<Parameter>>> Module::namedParameters() const {
    std::vector<std::pair<std::string, std::shared_ptr<Parameter>>> result;
    namedParametersImpl(result, "");
    return result;
}

void Module::namedParametersImpl(
    std::vector<std::pair<std::string, std::shared_ptr<Parameter>>>& result,
    const std::string& prefix) const {
    // Build hierarchical parameter names with dot notation (e.g., "layer1.weight")
    // Uses prefix accumulation during DFS traversal

    for (auto& [name, param] : mParameters) {
        const auto current_prefix = prefix.empty() ? name : prefix + "." + name;
        result.emplace_back(current_prefix, param);
    }

    for (auto& [name, submodule] : mSubmodules) {
        const auto new_prefix = prefix.empty() ? name : prefix + "." + name;
        submodule->namedParametersImpl(result, new_prefix);
    }
}

// ============================================================================
// Utility Methods
// ============================================================================

void Module::to(const Device& device) {
    // Move own parameters
    for (auto& [name, param] : mParameters) {
        param->to(device);
    }

    // Recursively move submodule parameters
    for (auto& [name, submodule] : mSubmodules) {
        submodule->to(device);
    }
}

void Module::zeroGrad() {
    // Zero own parameter gradients
    for (auto& [name, param] : mParameters) {
        param->zeroGrad();
    }

    // Recursively zero submodule gradients
    for (auto& [name, submodule] : mSubmodules) {
        submodule->zeroGrad();
    }
}

void Module::train(bool mode) {
    mTraining = mode;

    // Recursively set mode for submodules
    for (auto& [name, submodule] : mSubmodules) {
        submodule->train(mode);
    }
}

}  // namespace nn
}  // namespace loom
