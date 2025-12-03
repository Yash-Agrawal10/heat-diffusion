#include "kernels/kernels.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_fast(const double* __restrict__ u, double* __restrict__ u_new, const int N,
                                           const double lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = i * N * N + j * N + k;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
        u_new[idx] = u[idx] + lambda * (u[(i + 1) * N * N + j * N + k] + u[(i - 1) * N * N + j * N + k] +
                                        u[i * N * N + (j + 1) * N + k] + u[i * N * N + (j - 1) * N + k] +
                                        u[i * N * N + j * N + (k + 1)] + u[i * N * N + j * N + (k - 1)] - 6.0 * u[idx]);
    }
}