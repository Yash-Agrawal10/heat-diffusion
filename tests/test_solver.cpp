#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "helpers/analytic.hpp"
#include "solver.hpp"

TEST_CASE("test solver for N=10, T=0.1") {
    // Define constants
    const int N = 10;
    const double h = 1.0 / (N - 1);
    const double T = 0.1;
    const double epsilon = 1e-2;

    // Initialize temperature grid
    std::vector<std::vector<std::vector<double>>> u(N,
                                                    std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0)));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
                    u[i][j][k] = 0.0;
                } else {
                    u[i][j][k] = std::sin(M_PI * i * h) * std::sin(M_PI * j * h) * std::sin(M_PI * k * h);
                }
            }
        }
    }

    // Get solver solution
    heat_diffusion_3d(N, T, u);

    // Compare solver solution to analytic solution
    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * h;
                double y = j * h;
                double z = k * h;
                double u_exact = u_analytic(x, y, z, T);
                double error = std::abs(u[i][j][k] - u_exact);
                max_error = std::max(max_error, error);
            }
        }
    }

    CHECK(max_error < epsilon);
}