#pragma once

#include "problem_spec.hpp"

#include <chrono>
#include <iostream>
#include <vector>

using Kernel_t = std::vector<double> (*)(const Constants&, std::vector<double>&, std::vector<double>&, bool);

inline std::vector<double> heat_diffusion_solver(const ProblemSpec& spec, Kernel_t kernel, bool verbose = false) {
    // Define constants
    Constants consts = compute_constants(spec);
    const int N = consts.N;

#ifdef PROFILE
    std::cout << "# Problem\n"
              << "N=" << spec.N << "\n"
              << "T=" << spec.T << "\n";
#endif

    // Initialize grids
    auto u = initial_condition_unit_cube(N, spec.initial_condition);
    std::vector<double> u_new(N * N * N, 0.0);

    // Run kernel
#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    auto result = kernel(consts, u, u_new, verbose);
#ifdef PROFILE
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // Timing section created by kernel
    std::cout << "total=" << elapsed.count() << " s\n";
#endif
    return result;
}