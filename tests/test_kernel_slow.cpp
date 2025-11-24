#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "helpers/analytic.hpp"
#include "kernel_slow.hpp"
#include "problem_spec.hpp"
#include "solver.hpp"

TEST_CASE("test solver for N=32, T=0.1") {
    // Define constants
    const int N = 32;
    const double T = 0.1;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Get solutions
    auto u_numerical = heat_diffusion_solver(spec, heat_diffuision_kernel_slow);
    auto u_analytic = analytic_unit_cube(N, T);

    // Compare solver solution to analytic solution
    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int index = i * N * N + j * N + k;
                double error = std::abs(u_numerical[index] - u_analytic[index]);
                max_error = std::max(max_error, error);
            }
        }
    }

    CHECK(max_error < epsilon);
}