#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "helpers/analytic.hpp"
#include "helpers/compare.hpp"
#include "kernels.hpp"
#include "problem_spec.hpp"
#include "solver.hpp"

TEST_CASE("test slow kernel for N=32, T=0.1") {
    // Define constants
    const int N = 32;
    const double T = 0.1;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Get solutions
    auto u_numerical = heat_diffusion_solver(spec, heat_diffusion_kernel_slow);
    auto u_analytic = analytic_unit_cube(N, T);

    // Compare solver solution to analytic solution
    double max_error = get_max_error(u_numerical, u_analytic, N);
    CHECK(max_error < epsilon);
}

TEST_CASE("test slow kernel for N=64, T=0.05") {
    // Define constants
    const int N = 64;
    const double T = 0.05;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Get solutions
    auto u_numerical = heat_diffusion_solver(spec, heat_diffusion_kernel_slow);
    auto u_analytic = analytic_unit_cube(N, T);

    // Compare solver solution to analytic solution
    double max_error = get_max_error(u_numerical, u_analytic, N);
    CHECK(max_error < epsilon);
}

TEST_CASE("test slow kernel for N=100, T=0.1") {
    // Define constants
    const int N = 100;
    const double T = 0.1;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Get solutions
    auto u_numerical = heat_diffusion_solver(spec, heat_diffusion_kernel_slow);
    auto u_analytic = analytic_unit_cube(N, T);

    // Compare solver solution to analytic solution
    double max_error = get_max_error(u_numerical, u_analytic, N);
    CHECK(max_error < epsilon);
}

TEST_CASE("test slow kernel for N=64, T=1.0") {
    // Define constants
    const int N = 64;
    const double T = 1.0;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Get solutions
    auto u_numerical = heat_diffusion_solver(spec, heat_diffusion_kernel_slow);
    auto u_analytic = analytic_unit_cube(N, T);

    // Compare solver solution to analytic solution
    double max_error = get_max_error(u_numerical, u_analytic, N);
    CHECK(max_error < epsilon);
}