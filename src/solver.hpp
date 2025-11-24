#pragma once

#include "problem_spec.hpp"

#include <vector>

using Kernel_t = std::vector<double> (*)(const Constants&, std::vector<double>&, std::vector<double>&);

inline std::vector<double> heat_diffusion_solver(const ProblemSpec& spec, Kernel_t kernel) {
    // Define constants
    Constants consts = compute_constants(spec);
    const int N = consts.N;

    // Initialize grids
    auto u = initial_condition_unit_cube(N, spec.initial_condition);
    std::vector<double> u_new(N * N * N, 0.0);

    // Run kernel
    return kernel(consts, u, u_new);
}