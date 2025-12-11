#pragma once

namespace loom {
namespace autograd {

/**
 * @brief Global state for disabling autograd
 *
 * Thread-local state that controls whether gradient tracking is enabled.
 * Used by NoGrad RAII guard to temporarily disable autograd.
 */
class NoGradMode {
   public:
    /**
     * @brief Check if gradient tracking is currently disabled
     * @return true if NoGrad context is active, false otherwise
     */
    static bool isEnabled();

    /**
     * @brief Enable or disable gradient tracking
     * @param enabled true to disable gradients, false to enable
     */
    static void setEnabled(bool enabled);

   private:
    static thread_local bool sEnabled;
};

/**
 * @brief RAII guard to disable gradient tracking
 *
 * Usage:
 * @code
 *   {
 *       NoGrad no_grad;
 *       // Operations here won't build computation graph
 *       Tensor y = x * 2;  // No grad_fn attached
 *   }
 *   // Gradient tracking restored
 * @endcode
 *
 * Common use cases:
 * - Inference (no need to track gradients)
 * - Optimizer updates (W -= lr * grad shouldn't track)
 * - Data preprocessing
 */
class NoGrad {
   public:
    /**
     * @brief Disable gradient tracking for this scope
     *
     * Saves previous state and disables gradient tracking.
     * Previous state will be restored when object is destroyed.
     */
    NoGrad() : mPrevState(NoGradMode::isEnabled()) { NoGradMode::setEnabled(true); }

    /**
     * @brief Restore previous gradient tracking state
     */
    ~NoGrad() { NoGradMode::setEnabled(mPrevState); }

    // Prevent copying to avoid state confusion
    NoGrad(const NoGrad&) = delete;
    NoGrad& operator=(const NoGrad&) = delete;

   private:
    bool mPrevState;
};

}  // namespace autograd
}  // namespace loom
