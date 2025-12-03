#pragma once

#include <hip/hip_runtime.h>

#include <vector>

void kernel_slow(std::vector<double>& u, std::vector<double>& u_new, int N, double lambda);
__global__ void kernel_fast(const double* u, double* u_new, int N_x, int N_y, int N_z, double lambda);