#include "solvers/solvers.hpp"

#include "kernels/kernels.hpp"
#include "util/constants.hpp"
#include "util/helpers.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>

std::vector<double> solver_fast(const ProblemSpec& spec, Mode mode) {
    using Clock = std::chrono::high_resolution_clock;

    // Define constants
    Constants consts = compute_constants(spec);

    // Define GPU thread layout
    const dim3 blockSize(32, 4, 1);
    const dim3 gridSize((spec.N + blockSize.x - 1) / blockSize.x, (spec.N + blockSize.y - 1) / blockSize.y,
                        (spec.N + blockSize.z - 1) / blockSize.z);

    // Allocate and initialize host memory
    auto u = initial_condition_unit_cube(spec.N, spec.initial_condition);
    std::vector<double> u_new(spec.N * spec.N * spec.N, 0.0);

    // Allocate and initialize device memory
    auto device_alloc_start = Clock::now();
    double *d_u, *d_u_new;
    hipMalloc(&d_u, u.size() * sizeof(double));
    hipMalloc(&d_u_new, u_new.size() * sizeof(double));
    hipMemcpy(d_u, u.data(), u.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_u_new, u_new.data(), u_new.size() * sizeof(double), hipMemcpyHostToDevice);
    auto device_alloc_end = Clock::now();
    if (mode == Mode::profile) {
        std::chrono::duration<double> alloc_elapsed = device_alloc_end - device_alloc_start;
        std::cout << "Device memory allocation and initialization time: " << alloc_elapsed.count() << " seconds\n";
    }

    // Select mode
    if (mode == Mode::profile) {
        auto start = Clock::now();
        for (int i = 0; i < consts.n_steps; ++i) {
            kernel_fast<<<gridSize, blockSize>>>(d_u, d_u_new, spec.N, consts.lambda);
            std::swap(d_u, d_u_new);
        }
        auto end = Clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Kernel time: " << elapsed.count() << " seconds\n";
    } else if (mode == Mode::output) {
        const int num_frames = 60;
        const int output_interval = std::max(1, consts.n_steps / num_frames);
        dump_state(u, spec.N, 0);
        for (int i = 0; i < consts.n_steps; ++i) {
            kernel_fast<<<gridSize, blockSize>>>(d_u, d_u_new, spec.N, consts.lambda);
            std::swap(d_u, d_u_new);
            if ((i + 1) % output_interval == 0 && (i + 1) != consts.n_steps) {
                hipMemcpy(u.data(), d_u, u.size() * sizeof(double), hipMemcpyDeviceToHost);
                dump_state(u, spec.N, i + 1);
            }
        }
        hipMemcpy(u.data(), d_u, u.size() * sizeof(double), hipMemcpyDeviceToHost);
        dump_state(u, spec.N, consts.n_steps);
    } else if (mode == Mode::eval) {
        for (int step = 0; step < consts.n_steps; ++step) {
            kernel_fast<<<gridSize, blockSize>>>(d_u, d_u_new, spec.N, consts.lambda);
            std::swap(d_u, d_u_new);
        }
        hipMemcpy(u.data(), d_u, u.size() * sizeof(double), hipMemcpyDeviceToHost);
    }

    return u;
}