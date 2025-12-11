#include "loom/autograd/no_grad.h"

namespace loom {
namespace autograd {

// Thread-local state initialized to false (gradients enabled by default)
thread_local bool NoGradMode::sEnabled = false;

bool NoGradMode::isEnabled() {
    return sEnabled;
}

void NoGradMode::setEnabled(bool enabled) {
    sEnabled = enabled;
}

}  // namespace autograd
}  // namespace loom
