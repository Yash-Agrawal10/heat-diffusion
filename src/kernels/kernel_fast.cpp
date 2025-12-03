#include "kernels/kernels.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_fast(const double* __restrict__ u, double* __restrict__ u_new, const int N_x, const int N_y,
                            const int N_z, const double lambda) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    int idx = i * N_y_total * N_z_total + j * N_z_total + k;

    if (i >= 1 && i <= N_x && j >= 1 && j <= N_y && k >= 1 && k <= N_z) {
        u_new[idx] =
            u[idx] + lambda * (u[(i + 1) * x_stride + j * y_stride + k] + u[(i - 1) * x_stride + j * y_stride + k] +
                               u[i * x_stride + (j + 1) * y_stride + k] + u[i * x_stride + (j - 1) * y_stride + k] +
                               u[i * x_stride + j * y_stride + (k + 1)] + u[i * x_stride + j * y_stride + (k - 1)] -
                               6.0 * u[idx]);
    }
}