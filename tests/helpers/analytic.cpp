#include "analytic.hpp"

#include <vector>

std::vector<double> analytic_unit_cube(int N, double T) {
    std::vector<double> u(N * N * N);
    double h = 1.0 / (N - 1);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * h;
                double y = j * h;
                double z = k * h;
                u[i * N * N + j * N + k] = analytic_solution(x, y, z, T);
            }
        }
    }

    return u;
}