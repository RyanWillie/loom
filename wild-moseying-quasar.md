# Neural Network Module System Implementation Plan

## Overview

Implement a PyTorch-style neural network module system for the Loom framework to enable MNIST digit recognition training. The system follows existing codebase patterns (mCamelCase naming, lazy autograd initialization, shared_ptr semantics) and integrates with the existing autograd engine.

**Goal:** Train MNIST classifier achieving >90% accuracy

**Estimated Effort:** ~3,700 lines across 8 components + tests

---

## Implementation Phases

### Phase 1: Foundation (No Dependencies)

#### 1.1 Fix and Implement Parameter Class
**Files:**
- `include/loom/nn/parameter.h` (EXISTS - needs fixes)
- `src/common/nn/parameter.cpp` (CREATE)
- `tests/test_parameter.cpp` (CREATE)

**Critical Bugs to Fix:**
1. Lines 47, 52: `grad()` returns `Tensor&` but `Tensor::grad()` returns `std::shared_ptr<Tensor>`
   - **Fix:** Change return type to `std::shared_ptr<Tensor>`
2. Lines 68, 76, 91, 106, 116: Factory methods use `vector<int>` instead of `vector<size_t>`
   - **Fix:** Change all to `const std::vector<size_t>&` to match Tensor API

**Implementation (~150 lines):**
- Constructor: Call `mData.requiresGrad(requiresGrad)`
- `zeroGrad()`: Delegate to `mData.zeroGrad()`
- Factory methods:
  - `zeros/ones`: Create Tensor, wrap in Parameter, set requiresGrad
  - `kaiming`: `std = sqrt(2.0 / fan_in)`, multiply `Tensor::randn()` by std
  - `xavier`: `std = sqrt(1.0 / fan_in)`, multiply `Tensor::randn()` by std
  - `uniform`: Create zeros tensor, call `tensor.uniform(low, high)`
- `to()`: Return new Parameter with `mData.toDevice(device)`

**Tests (~200 lines):**
- Factory method correctness (shape, dtype, requiresGrad)
- Kaiming initialization variance validation
- Device movement
- Gradient operations

---

#### 1.2 Implement Functional Activations
**Files:**
- `include/loom/nn/functional.h` (CREATE)
- `src/common/nn/functional.cpp` (CREATE)
- `include/loom/autograd/nodes/activation_ops.h` (UPDATE - add new nodes)
- `src/common/autograd/nodes/activation_ops.cpp` (UPDATE - add implementations)
- `tests/test_functional.cpp` (CREATE)

**Activations to Add:**
1. **Sigmoid:** `1 / (1 + exp(-x))` with numerical stability for large |x|
2. **Tanh:** `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
3. **Softmax:** `exp(x - max(x)) / sum(exp(x - max(x)))` along dimension (log-sum-exp trick)
4. **LogSoftmax:** `x - max(x) - log(sum(exp(x - max(x))))` for numerical stability

**Backward Nodes (~200 lines total):**
- `SigmoidBackward`: `grad_input = gradOutput * y * (1 - y)` where y is saved output
- `TanhBackward`: `grad_input = gradOutput * (1 - y²)` where y is saved output
- `SoftmaxBackward`: `grad_input = y * (gradOutput - sum(gradOutput * y))` along dim
- `LogSoftmaxBackward`: Similar to softmax but for log probabilities

**Follow ReLUBackward Pattern:**
```cpp
class SigmoidBackward : public Node {
public:
    SigmoidBackward(const Tensor& output);
    std::vector<Tensor> backward(const Tensor& gradOutput) override;
private:
    Tensor mSavedOutput;  // Save sigmoid(x), not x
};
```

**Implementation Pattern (per activation, ~60 lines each):**
1. Compute forward pass element-wise
2. Check `x.requiresGrad() && !NoGradMode::isEnabled()`
3. Create backward node with saved tensors
4. Set `nextFunctions` if `x.gradFn()` exists
5. Set `inputTensors` for gradient accumulation
6. Attach to result: `result.setGradFn(node); result.requiresGrad(true)`

**Tests (~300 lines):**
- Forward pass numerical correctness
- Backward pass gradient checking (finite differences)
- Numerical stability edge cases
- Shape preservation
- Autograd integration

---

### Phase 2: Module System (Depends on Phase 1)

#### 2.1 Implement Module Base Class
**Files:**
- `include/loom/nn/module.h` (CREATE)
- `src/common/nn/module.cpp` (CREATE)
- `tests/test_module.cpp` (CREATE)

**Design (~350 lines total):**

**Header (~200 lines):**
```cpp
class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;

    // Explicit registration API
    std::shared_ptr<Parameter> registerParameter(const std::string& name, const Parameter& param);
    std::shared_ptr<Module> registerModule(const std::string& name, std::shared_ptr<Module> module);

    // Recursive parameter collection
    std::vector<std::shared_ptr<Parameter>> parameters() const;
    std::vector<std::pair<std::string, std::shared_ptr<Parameter>>> namedParameters() const;

    // Utilities
    void to(const Device& device);
    void zeroGrad();
    void train(bool mode = true) { mTraining = mode; }
    void eval() { train(false); }

protected:
    std::map<std::string, std::shared_ptr<Parameter>> mParameters;
    std::map<std::string, std::shared_ptr<Module>> mSubmodules;
    bool mTraining = true;
};
```

**Implementation (~150 lines):**
- `registerParameter`: Store in map, return shared_ptr (for member variable assignment)
- `registerModule`: Store in map, return shared_ptr
- `parameters()`: Collect own params + recursively collect from submodules
- `namedParameters()`: DFS traversal with prefix accumulation ("layer1.weight", etc.)
- `to()`: Move all params + recursively call on submodules
- `zeroGrad()`: Zero own params + recursively call on submodules

**Tests (~250 lines):**
- Parameter registration and retrieval
- Duplicate registration throws error
- Recursive parameter collection
- Named parameters with correct prefixes
- Device movement propagation
- Gradient zeroing propagation

---

#### 2.2 Implement Linear Layer
**Files:**
- `include/loom/nn/linear.h` (CREATE)
- `src/common/nn/linear.cpp` (CREATE)
- `tests/test_linear.cpp` (CREATE)

**Design (~180 lines total):**

**Constructor Pattern:**
```cpp
Linear::Linear(int inFeatures, int outFeatures, bool bias) {
    mWeight = registerParameter("weight",
        Parameter::kaiming({outFeatures, inFeatures}));
    if (bias) {
        mBias = registerParameter("bias",
            Parameter::zeros({outFeatures}));
    }
}
```

**Forward Pass:**
```cpp
Tensor forward(const Tensor& input) {
    // y = xW^T + b
    // input: [batch, in_features]
    // weight: [out_features, in_features]
    Tensor weight_t = mWeight->data().transpose();  // [in, out]
    Tensor output = input.matmul(weight_t);         // [batch, out]
    if (mBias) {
        output = output + mBias->data();  // Broadcasting
    }
    return output;
}
```

**Tests (~200 lines):**
- Construction with/without bias
- Shape validation (2D input required)
- Forward pass correctness
- Gradient flow through matmul and bias
- Batch processing

---

### Phase 3: Composition & Loss (Depends on Phase 2)

#### 3.1 Implement Sequential Container
**Files:**
- `include/loom/nn/sequential.h` (CREATE)
- `src/common/nn/sequential.cpp` (CREATE)
- `tests/test_sequential.cpp` (CREATE)

**Design (~180 lines total):**
- Constructor from `std::initializer_list<std::shared_ptr<Module>>`
- `add()` method for builder pattern
- `forward()`: Chain module calls `module_n(...module_1(input))`
- Auto-register modules as "layer0", "layer1", etc.

**Tests (~150 lines):**
- Construction and chaining
- Parameter collection from all layers
- Gradient flow through chain

---

#### 3.2 Implement CrossEntropyLoss
**Files:**
- `include/loom/nn/loss.h` (CREATE)
- `src/common/nn/loss.cpp` (CREATE)
- `tests/test_loss.cpp` (CREATE)

**Design (~300 lines total):**

**Base Loss Class:**
```cpp
class Loss : public Module {
public:
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets) = 0;
    Tensor forward(const Tensor&) override { throw ...; }  // Hide single-arg version
};
```

**CrossEntropyLoss Implementation:**
1. Compute `log_probs = logSoftmax(predictions, dim=-1)`
2. Index log_probs at target classes: `loss[i] = -log_probs[i, target[i]]`
3. Apply reduction (mean/sum/none)

**Critical:** Use log-sum-exp trick for numerical stability in softmax

**Tests (~200 lines):**
- Forward correctness vs hand-calculated values
- Reduction modes
- Gradient flow
- Shape validation

---

### Phase 4: Optimization (Depends on Phase 2)

#### 4.1 Implement Optimizer Base + SGD
**Files:**
- `include/loom/optim/optimizer.h` (CREATE)
- `include/loom/optim/sgd.h` (CREATE)
- `src/common/optim/sgd.cpp` (CREATE)
- `tests/test_optimizer.cpp` (CREATE)

**Design (~240 lines total):**

**Optimizer Base:**
```cpp
class Optimizer {
protected:
    std::vector<std::shared_ptr<nn::Parameter>> mParameters;
    double mLearningRate;
public:
    virtual void step() = 0;
    void zeroGrad();  // Call zeroGrad() on all parameters
};
```

**SGD Implementation:**
```cpp
void SGD::step() {
    autograd::NoGrad no_grad;  // CRITICAL: disable autograd!
    for (auto& param : mParameters) {
        if (param->grad()) {
            param->data() -= (*param->grad()) * mLearningRate;
        }
    }
}
```

**Why NoGrad:** Without it, `param -= lr * grad` would build a computation graph!

**Tests (~200 lines):**
- Parameter updates
- Learning rate application
- NoGrad context validation
- Gradient zeroing

---

### Phase 5: Integration & Training (Depends on All)

#### 5.1 Create MNIST Training Example
**Files:**
- `examples/mnist_train/main.cpp` (CREATE)
- `examples/mnist_train/CMakeLists.txt` (CREATE)

**MLP Model (~320 lines total):**
```cpp
class MNISTModel : public nn::Module {
    std::shared_ptr<nn::Linear> mLayer1;  // 784 -> 128
    std::shared_ptr<nn::Linear> mLayer2;  // 128 -> 10

    Tensor forward(const Tensor& x) override {
        auto h = nn::functional::relu(mLayer1->forward(x));
        return mLayer2->forward(h);
    }
};
```

**Training Loop:**
1. Load MNIST with DataLoader (batch_size=64, shuffle=true)
2. Create model, loss, optimizer
3. For each epoch:
   - For each batch:
     - Forward: `logits = model->forward(images)`
     - Loss: `loss = criterion->forward(logits, labels)`
     - Backward: `optimizer.zeroGrad(); loss.backward();`
     - Update: `optimizer.step()`
4. Compute accuracy: argmax predictions vs targets
5. Log metrics every N batches

**Expected Results:** >90% accuracy after 10 epochs with lr=0.01

---

## Critical Implementation Details

### 1. Kaiming Initialization Math
```cpp
// For weight [out_features, in_features]:
size_t fan_in = shape[0];  // First dimension
double std = sqrt(2.0 / fan_in);  // For ReLU networks
Tensor t = Tensor::randn(shape) * std;
```

### 2. Numerical Stability in Softmax
```cpp
// BAD: softmax = exp(x) / sum(exp(x))  → overflow for large x
// GOOD: Log-sum-exp trick
Tensor x_max = x.max(dim, /*keepdim=*/true);
Tensor x_shifted = x - x_max;  // Largest value is now 0
Tensor exp_x = /* element-wise exp(x_shifted) */;
Tensor softmax = exp_x / exp_x.sum(dim, /*keepdim=*/true);
```

### 3. Linear Layer Matrix Multiplication
```cpp
// Forward: y = xW^T + b
// x: [batch, in_features]
// W: [out_features, in_features]
// Need W^T: [in_features, out_features]
Tensor weight_t = weight.transpose();
Tensor output = input.matmul(weight_t);  // [batch, out] ✓
```

### 4. Optimizer NoGrad Context
```cpp
void SGD::step() {
    autograd::NoGrad no_grad;  // Must wrap parameter updates!
    // Without this, param -= lr * grad builds computation graph
    for (auto& param : mParameters) {
        param->data() -= (*param->grad()) * mLearningRate;
    }
}
```

### 5. Activation Backward Node Pattern
```cpp
// For sigmoid: save OUTPUT (not input) because derivative uses output
class SigmoidBackward : public Node {
    Tensor mSavedOutput;  // y = sigmoid(x)

    std::vector<Tensor> backward(const Tensor& gradOutput) override {
        // ∂L/∂x = ∂L/∂y * y * (1 - y)
        Tensor one = Tensor::ones(mSavedOutput.shape(), ...);
        return {gradOutput * mSavedOutput * (one - mSavedOutput)};
    }
};
```

---

## Testing Strategy

**Test Coverage per Component:**
1. Construction and initialization
2. Shape/dtype/device validation
3. Forward pass numerical correctness
4. Backward pass gradient checking (finite differences)
5. Edge cases (empty, zero-size, extreme values)
6. Autograd integration
7. Multi-batch processing

**Total Test Lines:** ~1,500 lines

---

## Build System Integration

**No CMakeLists.txt changes needed!**
- Root `CMakeLists.txt` uses `file(GLOB_RECURSE COMMON_SOURCES "src/common/*.cpp")`
- Automatically includes new `src/common/nn/*.cpp` and `src/common/optim/*.cpp`
- Tests use `file(GLOB_RECURSE TEST_SOURCES "*.cpp")` - automatically includes new tests

**Build Commands:**
```bash
./build.sh debug              # Development build
./build.sh test               # Build and run all tests
./build.sh release            # Optimized build for training
./build/examples/mnist_train  # Run training
```

---

## Implementation Order Summary

```
1. Fix Parameter.h bugs → Implement parameter.cpp → Test
2. Implement functional.h/cpp + backward nodes → Test
3. Implement Module base class → Test
4. Implement Linear layer → Test
5. Implement Sequential container → Test
6. Implement Loss classes → Test
7. Implement Optimizer base + SGD → Test
8. Create MNIST training example → Train and validate >90% accuracy
```

**Dependencies:**
- Parameter ← (none)
- Functional activations ← (none)
- Module ← Parameter
- Linear ← Module
- Sequential ← Module
- Loss ← Module, Functional
- Optimizer ← Parameter
- MNIST example ← All

---

## Success Criteria

✅ All tests pass (367 existing + ~1,500 new = ~1,867 total)
✅ MNIST training converges to >90% accuracy
✅ No memory leaks (validate with AddressSanitizer)
✅ Clean code (passes clang-tidy)
✅ Follows existing codebase patterns

---

## Estimated Line Counts

| Component | Header | Implementation | Tests | Total |
|-----------|--------|----------------|-------|-------|
| Parameter (fixes) | 0 (fix existing) | 150 | 200 | 350 |
| Functional activations | 60 | 450 | 300 | 810 |
| Module base | 200 | 150 | 250 | 600 |
| Linear layer | 80 | 100 | 200 | 380 |
| Sequential | 100 | 80 | 150 | 330 |
| Loss classes | 150 | 150 | 200 | 500 |
| Optimizer + SGD | 160 | 80 | 200 | 440 |
| MNIST example | - | 320 | - | 320 |
| **TOTAL** | **750** | **1,480** | **1,500** | **3,730** |

---

## Key Files to Create/Modify

**Fix:**
- `include/loom/nn/parameter.h` (fix grad() and factory method signatures)

**Create (Headers):**
- `include/loom/nn/module.h`
- `include/loom/nn/linear.h`
- `include/loom/nn/sequential.h`
- `include/loom/nn/functional.h`
- `include/loom/nn/loss.h`
- `include/loom/optim/optimizer.h`
- `include/loom/optim/sgd.h`

**Create (Implementation):**
- `src/common/nn/parameter.cpp`
- `src/common/nn/module.cpp`
- `src/common/nn/linear.cpp`
- `src/common/nn/sequential.cpp`
- `src/common/nn/functional.cpp`
- `src/common/nn/loss.cpp`
- `src/common/optim/sgd.cpp`

**Update:**
- `include/loom/autograd/nodes/activation_ops.h` (add 4 new backward nodes)
- `src/common/autograd/nodes/activation_ops.cpp` (implement backward nodes)

**Create (Tests):**
- `tests/test_parameter.cpp`
- `tests/test_module.cpp`
- `tests/test_linear.cpp`
- `tests/test_sequential.cpp`
- `tests/test_functional.cpp`
- `tests/test_loss.cpp`
- `tests/test_optimizer.cpp`

**Create (Example):**
- `examples/mnist_train/main.cpp`
- `examples/mnist_train/CMakeLists.txt`
