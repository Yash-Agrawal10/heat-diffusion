#include "kernel_slow.hpp"
#include "problem_spec.hpp"
#include "solver.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // Define problem
    const int N = 10;
    const double T = 0.1;
    auto initial_condition = [](double x, double y, double z) {
        return std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
    };
    ProblemSpec spec{ N, T, initial_condition };

    // Perform heat diffusion
    auto u = heat_diffusion_solver(spec, heat_diffuision_kernel_slow);

    // Print results at the center slice
    std::cout << "Temperature distribution at center slice (k = " << N / 2 << "):\n";
    int k = N / 2;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N * N + j * N + k;
            std::cout << u[idx] << " ";
        }
        std::cout << "\n";
    }
}