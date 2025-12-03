#include "solvers/solvers.hpp"

#include "kernels/kernels.hpp"
#include "util/constants.hpp"
#include "util/initial_condition.hpp"
#include "util/output.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>

void check_hip_error(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl;
        exit(-1);
    }
}

std::vector<double> solver_shared_gpu(const ProblemSpec& spec, Mode mode, bool verbose) {
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

    // Main time-stepping loop
    auto start = Clock::now();
    const int num_frames = 60;
    const int output_interval = std::max(1, consts.n_steps / num_frames);

    if (mode == Mode::output) {
        dump_state(u, spec.N, 0);
    }
    if (verbose) {
        std::cout << "Starting shared GPU solver with " << consts.n_steps << " time steps...\n";
    }

    for (int step = 0; step < consts.n_steps; ++step) {
        kernel_shared_gpu<<<gridSize, blockSize>>>(d_u, d_u_new, spec.N, consts.lambda);
        std::swap(d_u, d_u_new);

        if (mode == Mode::output && (step + 1) % output_interval == 0 && (step + 1) != consts.n_steps) {
            hipMemcpy(u.data(), d_u, u.size() * sizeof(double), hipMemcpyDeviceToHost);
            dump_state(u, spec.N, step + 1);
        }
        if (verbose && (step + 1) % (consts.n_steps / 100) == 0) {
            std::cout << "Completed " << (step + 1) << " / " << consts.n_steps << " steps...\n";
        }
    }

    check_hip_error(hipDeviceSynchronize());
    hipMemcpy(u.data(), d_u, u.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_u);
    hipFree(d_u_new);

    if (mode == Mode::output) {
        dump_state(u, spec.N, consts.n_steps);
    }
    if (verbose) {
        std::cout << "Finished shared GPU solver.\n" << std::endl;
    }
    auto end = Clock::now();
    if (mode == Mode::profile) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n" << std::endl;
    }

    return u;
}