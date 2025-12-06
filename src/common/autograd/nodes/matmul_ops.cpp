#include "loom/autograd/nodes/matmul_ops.h"

#include "loom/tensor/tensor.h"

namespace loom {
namespace autograd {

// ============================================================================
// MatmulBackward: C = A @ B
// ============================================================================

MatmulBackward::MatmulBackward(const Tensor& a, const Tensor& b)
    : mSavedA(a), mSavedB(b) {
    mName = "MatmulBackward";
}

std::vector<Tensor> MatmulBackward::backward(const Tensor& gradOutput) {
    // Check that saved tensors haven't been modified in-place
    checkVersions();

    // Mathematical formulas:
    // Given: C = A @ B
    //        gradOutput = dL/dC  (gradient flowing back from output)
    // We need to compute:
    //   grad_a = dL/dA = (dL/dC) @ B^T
    //   grad_b = dL/dB = A^T @ (dL/dC)
    // Algorithm:
    // 1. Compute grad_a:
    //    a. Get B transpose: Use mSavedB.transpose()
    //    b. Multiply gradOutput @ B^T: Use gradOutput.matmul(B_transpose)
    // 2. Compute grad_b:
    //    a. Get A transpose: Use mSavedA.transpose()
    //    b. Multiply A^T @ gradOutput: Use A_transpose.matmul(gradOutput)
    // 3. Return vector containing {grad_a, grad_b}
    // Hint: The transpose() method swaps the last two dimensions
    // Hint: For 2D tensors [M,K], transpose gives [K,M]
    //
    // Shape verification (for debugging):
    //   If A:[M,K], B:[K,N], C:[M,N], gradOutput:[M,N]
    //   - B^T: [N,K]
    //   - grad_a = [M,N] @ [N,K] = [M,K] ✓ (matches A's shape)
    //   - A^T: [K,M]
    //   - grad_b = [K,M] @ [M,N] = [K,N] ✓ (matches B's shape)
    // Try alternative formula order based on matrix dimensions
    Tensor B_transpose = mSavedB.transpose();
    Tensor A_transpose = mSavedA.transpose();

    // For C = A @ B, the gradients are:
    // grad_a = gradOutput @ B^T
    // grad_b = A^T @ gradOutput
    Tensor grad_a = gradOutput.matmul(B_transpose);
    Tensor grad_b = A_transpose.matmul(gradOutput);

    return {grad_a, grad_b};
}

}  // namespace autograd
}  // namespace loom
