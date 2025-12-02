#include "kernel_slow.hpp"

#include "problem_spec.hpp"

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void heat_diffusion_kernel_step(const double* __restrict__ u, double* __restrict__ u_new, const int N,
                                           const double lambda) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = i * N * N + j * N + k;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
        u_new[idx] = u[idx] + lambda * (u[(i + 1) * N * N + j * N + k] + u[(i - 1) * N * N + j * N + k] +
                                        u[i * N * N + (j + 1) * N + k] + u[i * N * N + (j - 1) * N + k] +
                                        u[i * N * N + j * N + (k + 1)] + u[i * N * N + j * N + (k - 1)] - 6.0 * u[idx]);
    }
}

std::vector<double> heat_diffusion_kernel_fast(const Constants& consts, std::vector<double>& u,
                                               std::vector<double>& u_new, bool verbose) {
    const int N = consts.N;
    const dim3 blockSize(32, 4, 1);
    const dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y,
                        (N + blockSize.z - 1) / blockSize.z);
#ifdef PROFILE
    (void)verbose; // Suppress unused variable warning
#endif

#ifndef PROFILE
    const int total_outputs = 60;
    const int output_interval = std::max(1, consts.n_steps / total_outputs);
#endif

#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif

#ifndef PROFILE
    if (verbose) {
        dump_state(u, N, 0);
    }
#endif

    double* d_u;
    double* d_u_new;
    hipMalloc(&d_u, u.size() * sizeof(double));
    hipMalloc(&d_u_new, u_new.size() * sizeof(double));
    hipMemcpy(d_u, u.data(), u.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_u_new, u_new.data(), u_new.size() * sizeof(double), hipMemcpyHostToDevice);

    for (int n = 0; n < consts.n_steps; ++n) {
        heat_diffusion_kernel_step<<<gridSize, blockSize>>>(d_u, d_u_new, N, consts.lambda);
        hipDeviceSynchronize();
        std::swap(d_u, d_u_new);

#ifndef PROFILE
        hipMemcpy(u.data(), d_u, u.size() * sizeof(double), hipMemcpyDeviceToHost);
        int step = n + 1;
        if (verbose && step != consts.n_steps && step % output_interval == 0) {
            dump_state(u, N, step);
        }
#endif
    }

    hipMemcpy(u.data(), d_u, u.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_u);
    hipFree(d_u_new);

#ifdef PROFILE
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "# Timing\n"
              << "kernel_compute=" << elapsed.count() << " s\n";
#endif

#ifndef PROFILE
    if (verbose) {
        dump_state(u, N, consts.n_steps);
    }
#endif

    return u;
}