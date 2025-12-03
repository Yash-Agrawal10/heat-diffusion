#include "kernels/kernels.hpp"

#include <hip/hip_runtime.h>

__global__ void pack_x(const double* __restrict__ u, double* __restrict__ buf, const int N_x, const int N_y,
                       const int N_z, FaceSide side) {
    int i = (side == FaceSide::minus) ? 1 : N_x;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    if (j >= 1 && j <= N_y && k >= 1 && k <= N_z) {
        int u_idx = i * x_stride + j * y_stride + k;
        int buf_idx = (j - 1) * N_z + (k - 1);
        buf[buf_idx] = u[u_idx];
    }
}

__global__ void pack_y(const double* __restrict__ u, double* __restrict__ buf, const int N_x, const int N_y,
                       const int N_z, FaceSide side) {
    int j = (side == FaceSide::minus) ? 1 : N_y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    if (i >= 1 && i <= N_x && k >= 1 && k <= N_z) {
        int u_idx = i * x_stride + j * y_stride + k;
        int buf_idx = (i - 1) * N_z + (k - 1);
        buf[buf_idx] = u[u_idx];
    }
}

__global__ void pack_z(const double* __restrict__ u, double* __restrict__ buf, const int N_x, const int N_y,
                       const int N_z, FaceSide side) {
    int k = (side == FaceSide::minus) ? 1 : N_z;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    if (i >= 1 && i <= N_x && j >= 1 && j <= N_y) {
        int u_idx = i * x_stride + j * y_stride + k;
        int buf_idx = (i - 1) * N_y + (j - 1);
        buf[buf_idx] = u[u_idx];
    }
}

__global__ void unpack_x(const double* __restrict__ buf, double* __restrict__ u, const int N_x, const int N_y,
                         const int N_z, FaceSide side) {
    int i = (side == FaceSide::minus) ? 0 : N_x + 1;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    if (j >= 1 && j <= N_y && k >= 1 && k <= N_z) {
        int u_idx = i * x_stride + j * y_stride + k;
        int buf_idx = (j - 1) * N_z + (k - 1);
        u[u_idx] = buf[buf_idx];
    }
}

__global__ void unpack_y(const double* __restrict__ buf, double* __restrict__ u, const int N_x, const int N_y,
                         const int N_z, FaceSide side) {
    int j = (side == FaceSide::minus) ? 0 : N_y + 1;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    if (i >= 1 && i <= N_x && k >= 1 && k <= N_z) {
        int u_idx = i * x_stride + j * y_stride + k;
        int buf_idx = (i - 1) * N_z + (k - 1);
        u[u_idx] = buf[buf_idx];
    }
}

__global__ void unpack_z(const double* __restrict__ buf, double* __restrict__ u, const int N_x, const int N_y,
                         const int N_z, FaceSide side) {
    int k = (side == FaceSide::minus) ? 0 : N_z + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    if (i >= 1 && i <= N_x && j >= 1 && j <= N_y) {
        int u_idx = i * x_stride + j * y_stride + k;
        int buf_idx = (i - 1) * N_y + (j - 1);
        u[u_idx] = buf[buf_idx];
    }
}

__global__ void kernel_distributed_gpu(const double* __restrict__ u, double* __restrict__ u_new, const int N,
                                       const int N_x, const int N_y, const int N_z, const int i_start,
                                       const int j_start, const int k_start, const double lambda) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int k_global = k + k_start - 1;
    int j_global = j + j_start - 1;
    int i_global = i + i_start - 1;

    int N_y_total = N_y + 2;
    int N_z_total = N_z + 2;
    int x_stride = N_y_total * N_z_total;
    int y_stride = N_z_total;

    int idx = i * N_y_total * N_z_total + j * N_z_total + k;

    if (i >= 1 && i <= N_x && j >= 1 && j <= N_y && k >= 1 && k <= N_z && i_global > 0 && i_global < N - 1 &&
        j_global > 0 && j_global < N - 1 && k_global > 0 && k_global < N - 1) {
        u_new[idx] =
            u[idx] + lambda * (u[(i + 1) * x_stride + j * y_stride + k] + u[(i - 1) * x_stride + j * y_stride + k] +
                               u[i * x_stride + (j + 1) * y_stride + k] + u[i * x_stride + (j - 1) * y_stride + k] +
                               u[i * x_stride + j * y_stride + (k + 1)] + u[i * x_stride + j * y_stride + (k - 1)] -
                               6.0 * u[idx]);
    }
}