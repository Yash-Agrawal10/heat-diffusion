#pragma once

#include <hip/hip_runtime.h>

#include <vector>

void kernel_cpu(std::vector<double>& u, std::vector<double>& u_new, int N, double lambda);

__global__ void kernel_shared_gpu(const double* u, double* u_new, int N, double lambda);

__global__ void kernel_distributed_gpu(const double* u, double* u_new, int N, int N_x, int N_y, int N_z, int i_start,
                                       int j_start, int k_start, double lambda);

enum class FaceSide { minus, plus };

__global__ void pack_x(const double* u, double* buf, int N_x, int N_y, int N_z, FaceSide side);
__global__ void pack_y(const double* u, double* buf, int N_x, int N_y, int N_z, FaceSide side);
__global__ void pack_z(const double* u, double* buf, int N_x, int N_y, int N_z, FaceSide side);
__global__ void unpack_x(const double* buf, double* u, int N_x, int N_y, int N_z, FaceSide side);
__global__ void unpack_y(const double* buf, double* u, int N_x, int N_y, int N_z, FaceSide side);
__global__ void unpack_z(const double* buf, double* u, int N_x, int N_y, int N_z, FaceSide side);