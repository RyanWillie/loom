# Phase 4: Optimizer Implementation Plan

## Overview

Implement PyTorch-style optimizers to enable neural network training in the Loom framework. This phase focuses on creating the optimizer base class and SGD (Stochastic Gradient Descent) implementation.

**Goal:** Enable end-to-end MNIST training by implementing parameter update logic

**Prerequisites:**
- ✅ Phase 1-3 complete (Parameter, Module, Linear, Sequential, Loss all implemented)
- ✅ Autograd engine functional with backward pass
- ✅ 595/609 tests passing

**Estimated Effort:** ~240 lines of implementation + ~200 lines of tests

---

## Architecture Overview

### Optimizer Responsibilities

1. **Store references to parameters** - Track all learnable parameters in the model
2. **Update parameters** - Apply gradient-based updates (w = w - lr * grad)
3. **Zero gradients** - Reset accumulated gradients before each backward pass
4. **Disable autograd during updates** - Critical: use NoGrad context to prevent building computation graphs during parameter updates

### Key Design Pattern: NoGrad Context

**CRITICAL:** Parameter updates MUST happen in a `NoGrad` context:

```cpp
void SGD::step() {
    autograd::NoGrad no_grad;  // Disable autograd for this scope
    for (auto& param : mParameters) {
        if (param->grad()) {
            // Without NoGrad, this would create a new computation graph!
            param->data() -= (*param->grad()) * mLearningRate;
        }
    }
}
```

**Why?** The operation `param -= lr * grad` involves tensor arithmetic. Without `NoGrad`:
- The subtraction would create a `SubBackward` node
- Parameters would become non-leaf tensors
- The computation graph would grow unbounded across training iterations
- Memory would explode

---

## File Structure

### Files to Create

```
include/loom/optim/
├── optimizer.h          # Abstract base class (CREATE)
└── sgd.h               # SGD optimizer (CREATE)

src/common/optim/
└── sgd.cpp             # SGD implementation (CREATE)

tests/
└── test_optimizer.cpp  # Optimizer tests (CREATE)
```

**Note:** The root `CMakeLists.txt` uses `file(GLOB_RECURSE ...)` so new files are automatically included.

---

## Implementation Details

### 1. Optimizer Base Class

**File:** `include/loom/optim/optimizer.h`

**Purpose:** Abstract interface for all optimizers

**Design (~80 lines):**

```cpp
#pragma once

#include "loom/nn/parameter.h"
#include <memory>
#include <vector>

namespace loom {
namespace optim {

/**
 * @brief Abstract base class for all optimizers.
 *
 * Optimizers update model parameters based on their gradients. The typical
 * training loop pattern is:
 *
 *   optimizer.zeroGrad();      // Clear previous gradients
 *   loss = model.forward(...); // Forward pass
 *   loss.backward();           // Compute gradients
 *   optimizer.step();          // Update parameters
 *
 * All parameter updates MUST occur within a NoGrad context to prevent
 * building computation graphs during optimization.
 */
class Optimizer {
  protected:
    std::vector<std::shared_ptr<nn::Parameter>> mParameters;
    double mLearningRate;

  public:
    /**
     * @brief Construct optimizer with given parameters and learning rate.
     * @param parameters Vector of learnable parameters to optimize
     * @param lr Learning rate (step size)
     */
    Optimizer(const std::vector<std::shared_ptr<nn::Parameter>>& parameters,
              double lr);

    virtual ~Optimizer() = default;

    /**
     * @brief Perform a single optimization step (parameter update).
     *
     * Must be implemented by derived classes. Implementation MUST use
     * NoGrad context to disable autograd during parameter updates.
     */
    virtual void step() = 0;

    /**
     * @brief Zero out all parameter gradients.
     *
     * This should be called before each backward pass to clear accumulated
     * gradients from the previous iteration.
     */
    void zeroGrad();

    /**
     * @brief Get the current learning rate.
     */
    double learningRate() const { return mLearningRate; }

    /**
     * @brief Set a new learning rate (for learning rate scheduling).
     */
    void setLearningRate(double lr) { mLearningRate = lr; }

    /**
     * @brief Get the number of parameters being optimized.
     */
    size_t numParameters() const { return mParameters.size(); }
};

}  // namespace optim
}  // namespace loom
```

**Implementation:** `src/common/optim/optimizer.cpp` (if needed for base class methods)

```cpp
#include "loom/optim/optimizer.h"

namespace loom {
namespace optim {

Optimizer::Optimizer(const std::vector<std::shared_ptr<nn::Parameter>>& parameters,
                     double lr)
    : mParameters(parameters), mLearningRate(lr) {}

void Optimizer::zeroGrad() {
    for (auto& param : mParameters) {
        param->zeroGrad();
    }
}

}  // namespace optim
}  // namespace loom
```

---

### 2. SGD Optimizer

**File:** `include/loom/optim/sgd.h`

**Purpose:** Stochastic Gradient Descent optimizer

**Algorithm:**
```
For each parameter p:
    p = p - learning_rate * grad(p)
```

**Design (~60 lines):**

```cpp
#pragma once

#include "loom/optim/optimizer.h"

namespace loom {
namespace optim {

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 *
 * Implements vanilla SGD parameter updates:
 *   θ_new = θ_old - lr * ∇θ
 *
 * Where:
 *   θ     = parameter
 *   lr    = learning rate
 *   ∇θ    = gradient of loss with respect to parameter
 *
 * Example usage:
 *   auto params = model->parameters();
 *   SGD optimizer(params, 0.01);  // lr = 0.01
 *
 *   for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *       optimizer.zeroGrad();
 *       Tensor loss = loss_fn(model->forward(x), y);
 *       loss.backward();
 *       optimizer.step();
 *   }
 */
class SGD : public Optimizer {
  public:
    /**
     * @brief Construct SGD optimizer.
     * @param parameters Parameters to optimize
     * @param lr Learning rate (default: 0.01)
     */
    SGD(const std::vector<std::shared_ptr<nn::Parameter>>& parameters,
        double lr = 0.01);

    /**
     * @brief Perform SGD update step.
     *
     * Updates each parameter: p = p - lr * grad(p)
     * Uses NoGrad context to prevent building computation graphs.
     */
    void step() override;
};

}  // namespace optim
}  // namespace loom
```

**Implementation:** `src/common/optim/sgd.cpp`

```cpp
#include "loom/optim/sgd.h"
#include "loom/autograd/no_grad.h"

namespace loom {
namespace optim {

SGD::SGD(const std::vector<std::shared_ptr<nn::Parameter>>& parameters, double lr)
    : Optimizer(parameters, lr) {}

void SGD::step() {
    // CRITICAL: Wrap all parameter updates in NoGrad context
    // Without this, the subtraction operation would create backward nodes!
    autograd::NoGrad no_grad;

    for (auto& param : mParameters) {
        // Skip parameters without gradients (e.g., not used in forward pass)
        if (!param->grad()) {
            continue;
        }

        // Vanilla SGD update: θ = θ - lr * ∇θ
        // This is equivalent to: param->data() = param->data() - (*param->grad()) * mLearningRate
        param->data() -= (*param->grad()) * mLearningRate;
    }
}

}  // namespace optim
}  // namespace loom
```

---

## Critical Implementation Details

### 1. Why NoGrad Context is Essential

Without `NoGrad`, parameter updates create new computation graphs:

```cpp
// BAD - Creates computation graph during optimization!
void SGD::step() {
    for (auto& param : mParameters) {
        param->data() -= (*param->grad()) * mLearningRate;
        // This creates:
        // - MulBackward node for (*param->grad()) * mLearningRate
        // - SubBackward node for param->data() - ...
        // - param->data() becomes a NON-LEAF tensor
    }
}

// GOOD - Disables autograd during optimization
void SGD::step() {
    autograd::NoGrad no_grad;  // ← This is CRITICAL
    for (auto& param : mParameters) {
        param->data() -= (*param->grad()) * mLearningRate;
        // No computation graph created
        // param->data() remains a leaf tensor
    }
}
```

### 2. Gradient Existence Check

Always check if gradients exist before accessing:

```cpp
for (auto& param : mParameters) {
    if (!param->grad()) {
        continue;  // Skip parameters without gradients
    }
    // Safe to access param->grad() now
}
```

**Why?** Parameters may not have gradients if:
- They weren't used in the forward pass
- The backward pass hasn't been called yet
- Gradients were zeroed

### 3. In-Place Updates

Use in-place operators (`-=`, `+=`, `*=`) for efficiency:

```cpp
// GOOD - In-place update (no extra allocation)
param->data() -= (*param->grad()) * mLearningRate;

// BAD - Creates new tensor, requires reassignment
param->data() = param->data() - (*param->grad()) * mLearningRate;
```

### 4. Learning Rate as Double

Learning rates are typically small (0.001 to 0.1), so use `double`:

```cpp
double mLearningRate;  // GOOD - sufficient precision
// float mLearningRate;  // BAD - may lose precision for small LR
```

---

## Testing Strategy

**File:** `tests/test_optimizer.cpp`

**Test Coverage (~200 lines):**

### Test 1: SGD Construction
```cpp
TEST_F(OptimizerTest, SGDConstruction) {
    auto params = createTestParameters();  // Helper to create dummy params
    SGD optimizer(params, 0.01);

    EXPECT_EQ(optimizer.learningRate(), 0.01);
    EXPECT_EQ(optimizer.numParameters(), params.size());
}
```

### Test 2: Zero Gradients
```cpp
TEST_F(OptimizerTest, ZeroGradClearsAllGradients) {
    // Create parameters with gradients
    auto params = createTestParameters();
    for (auto& p : params) {
        // Simulate gradient computation
        p->data().backward(Tensor::ones(p->shape(), ...));
        ASSERT_NE(p->grad(), nullptr);  // Gradients exist
    }

    // Zero gradients
    SGD optimizer(params, 0.01);
    optimizer.zeroGrad();

    // Verify all gradients are zero
    for (auto& p : params) {
        ASSERT_NE(p->grad(), nullptr);  // Gradient tensor exists
        EXPECT_TRUE(isAllZeros(p->grad()));  // But values are zero
    }
}
```

### Test 3: Parameter Update
```cpp
TEST_F(OptimizerTest, SGDUpdatesParameters) {
    // Create simple 1D parameter
    Tensor data = Tensor::ones({3}, DType::FLOAT32, cpu_device);
    data.requiresGrad(true);
    auto param = std::make_shared<Parameter>(data);

    // Manually set gradient (normally computed by backward)
    Tensor grad = Tensor::full({3}, 2.0, DType::FLOAT32, cpu_device);
    param->data().backward(grad);

    // Store original values
    auto original = param->data().clone();

    // Perform SGD step with lr=0.1
    SGD optimizer({param}, 0.1);
    optimizer.step();

    // Check update: new = old - lr * grad = 1.0 - 0.1 * 2.0 = 0.8
    auto updated_acc = param->data().accessor<float, 1>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(updated_acc[i], 0.8f);
    }
}
```

### Test 4: NoGrad Context Verification
```cpp
TEST_F(OptimizerTest, StepDoesNotCreateComputationGraph) {
    Tensor data = Tensor::ones({2}, DType::FLOAT32, cpu_device);
    data.requiresGrad(true);
    auto param = std::make_shared<Parameter>(data);

    // Set gradient
    param->data().backward(Tensor::ones({2}, ...));

    // Verify param is a leaf before step
    EXPECT_TRUE(param->data().isLeaf());

    // Perform step
    SGD optimizer({param}, 0.01);
    optimizer.step();

    // Verify param is STILL a leaf (no graph created)
    EXPECT_TRUE(param->data().isLeaf());
    EXPECT_EQ(param->data().gradFn(), nullptr);
}
```

### Test 5: Multiple Parameters
```cpp
TEST_F(OptimizerTest, SGDUpdatesMultipleParameters) {
    // Create multiple parameters with different shapes
    auto p1 = std::make_shared<Parameter>(Tensor::ones({2, 3}, ...));
    auto p2 = std::make_shared<Parameter>(Tensor::ones({5}, ...));

    // Set gradients
    p1->data().backward(Tensor::full({2, 3}, 1.0, ...));
    p2->data().backward(Tensor::full({5}, 2.0, ...));

    // Update both parameters
    SGD optimizer({p1, p2}, 0.1);
    optimizer.step();

    // Verify both updated correctly
    // p1: 1.0 - 0.1 * 1.0 = 0.9
    // p2: 1.0 - 0.1 * 2.0 = 0.8
    // ... assertions ...
}
```

### Test 6: Skips Parameters Without Gradients
```cpp
TEST_F(OptimizerTest, SkipsParametersWithoutGradients) {
    auto p1 = std::make_shared<Parameter>(Tensor::ones({3}, ...));
    auto p2 = std::make_shared<Parameter>(Tensor::ones({3}, ...));

    // Only p1 has gradient
    p1->data().backward(Tensor::ones({3}, ...));
    // p2 has no gradient

    auto original_p2 = p2->data().clone();

    SGD optimizer({p1, p2}, 0.1);
    optimizer.step();  // Should not crash

    // p2 should be unchanged (no gradient to apply)
    EXPECT_TRUE(tensorEqual(p2->data(), original_p2));
}
```

### Test 7: Learning Rate Modification
```cpp
TEST_F(OptimizerTest, LearningRateCanBeModified) {
    auto param = std::make_shared<Parameter>(Tensor::ones({2}, ...));
    SGD optimizer({param}, 0.01);

    EXPECT_EQ(optimizer.learningRate(), 0.01);

    optimizer.setLearningRate(0.001);
    EXPECT_EQ(optimizer.learningRate(), 0.001);

    // Verify new LR is used in updates
    param->data().backward(Tensor::ones({2}, ...));
    optimizer.step();
    // Check updated values use lr=0.001
}
```

### Test 8: Integration Test with Model
```cpp
TEST_F(OptimizerTest, IntegrationWithLinearLayer) {
    // Create a simple linear layer
    auto linear = std::make_shared<nn::Linear>(5, 3);
    auto params = linear->parameters();

    // Create optimizer
    SGD optimizer(params, 0.01);

    // Simulate training iteration
    Tensor input = Tensor::rand({2, 5}, ...);  // Batch of 2
    Tensor target = Tensor::zeros({2, 3}, ...);

    // Forward pass
    Tensor output = linear->forward(input);

    // Simple loss (just sum for testing)
    Tensor loss = output.sum();

    // Backward pass
    optimizer.zeroGrad();
    loss.backward();

    // Verify gradients exist
    for (auto& p : params) {
        ASSERT_NE(p->grad(), nullptr);
    }

    // Parameter update
    auto weight_before = linear->parameters()[0]->data().clone();
    optimizer.step();
    auto weight_after = linear->parameters()[0]->data();

    // Verify weights changed
    EXPECT_FALSE(tensorEqual(weight_before, weight_after));
}
```

---

## Integration with Existing Code

### Using Optimizer in Training Loop

```cpp
#include "loom/nn/linear.h"
#include "loom/nn/sequential.h"
#include "loom/nn/loss.h"
#include "loom/optim/sgd.h"

// Create model
auto model = std::make_shared<nn::Sequential>();
model->add(std::make_shared<nn::Linear>(784, 128));
// ... add more layers

// Create optimizer
auto params = model->parameters();
optim::SGD optimizer(params, 0.01);

// Create loss function
auto criterion = std::make_shared<nn::CrossEntropyLoss>();

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        // Zero gradients from previous iteration
        optimizer.zeroGrad();

        // Forward pass
        Tensor output = model->forward(batch.data);
        Tensor loss = criterion->forward(output, batch.targets);

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step();
    }
}
```

---

## Build Integration

**No CMake changes needed!** The existing `file(GLOB_RECURSE ...)` pattern automatically includes:
- `src/common/optim/*.cpp`
- `tests/test_optimizer.cpp`

**Build commands:**
```bash
./build.sh debug    # Development build
./build.sh test     # Build and run tests
```

---

## Success Criteria

✅ All optimizer tests pass (8 tests)
✅ Integration test with Linear layer passes
✅ No memory leaks (verify with AddressSanitizer)
✅ Parameters remain leaf tensors after `step()`
✅ Total test count increases from 609 to ~617 tests
✅ All existing tests still pass (595+ passing)

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Forgetting NoGrad Context
```cpp
// WRONG - Creates computation graphs
void step() {
    for (auto& param : mParameters) {
        param->data() -= (*param->grad()) * mLearningRate;
    }
}
```

### ❌ Pitfall 2: Not Checking Gradient Existence
```cpp
// WRONG - May crash if grad is nullptr
void step() {
    for (auto& param : mParameters) {
        param->data() -= (*param->grad()) * mLearningRate;  // Crash!
    }
}
```

### ❌ Pitfall 3: Using Float for Learning Rate
```cpp
// WRONG - Loses precision for small learning rates
float mLearningRate;  // 0.0001 may become 0.0001001 or 0.0000999

// RIGHT
double mLearningRate;
```

### ❌ Pitfall 4: Modifying Gradients During Step
```cpp
// WRONG - Don't modify gradients during step
void step() {
    for (auto& param : mParameters) {
        auto& grad = param->grad();
        *grad *= 0.9;  // BAD - modifying gradient
        param->data() -= *grad * mLearningRate;
    }
}
```

---

## Next Steps After Phase 4

Once optimizers are implemented, you can:

1. **Phase 5:** Create MNIST training example
2. **Verify end-to-end training works**
3. **Achieve >90% accuracy on MNIST**

The optimizer is the final missing piece to enable neural network training!

---

## Reference Implementation Pattern

Follow the same patterns used in existing Loom code:

1. **Naming:** mCamelCase for members, PascalCase for classes
2. **Includes:** Group by category (STL, loom/autograd, loom/nn)
3. **Namespacing:** `loom::optim` namespace
4. **Documentation:** Doxygen-style comments with @brief, @param
5. **Testing:** One test file per component, use GoogleTest
6. **Error handling:** Check for nullptr, validate inputs

---

## Estimated Line Counts

| Component | Lines | File |
|-----------|-------|------|
| Optimizer.h | 80 | include/loom/optim/optimizer.h |
| Optimizer.cpp | 20 | src/common/optim/optimizer.cpp |
| SGD.h | 60 | include/loom/optim/sgd.h |
| SGD.cpp | 40 | src/common/optim/sgd.cpp |
| Tests | 200 | tests/test_optimizer.cpp |
| **TOTAL** | **400** | 5 files |

---

## Quick Start Checklist

- [ ] Create `include/loom/optim/optimizer.h`
- [ ] Create `include/loom/optim/sgd.h`
- [ ] Create `src/common/optim/sgd.cpp`
- [ ] Create `tests/test_optimizer.cpp`
- [ ] Build: `./build.sh debug`
- [ ] Test: `./build.sh test`
- [ ] Verify: 8 new tests passing, 603+ total tests passing
- [ ] Verify: NoGrad context used in `step()`
- [ ] Verify: Parameters remain leaf tensors after updates

---

**Ready to implement!** This plan provides all the details needed to implement Phase 4 optimizers and enable MNIST training.
