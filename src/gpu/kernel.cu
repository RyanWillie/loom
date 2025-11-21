#include <stdio.h>

#include "gpu/kernel.h"

__global__ void dummy_kernel() {
    // Do nothing
}

void call_cuda_kernel() {
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("CUDA kernel executed.\n");
}
