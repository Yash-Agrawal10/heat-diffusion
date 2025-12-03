#include "analytic.hpp"

#include "util/decomposition.hpp"

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

std::vector<double> analytic_distributed(int N, Decomposition& decomp, double T) {
    std::vector<double> u((decomp.N_x + 2) * (decomp.N_y + 2) * (decomp.N_z + 2), 0.0);
    const double h = 1.0 / (N - 1);

    auto get_index = [=](int i, int j, int k) {
        return i * (decomp.N_y + 2) * (decomp.N_z + 2) + j * (decomp.N_z + 2) + k;
    };

    for (int i = 1; i < decomp.N_x + 1; ++i) {
        int global_i = decomp.i_start + i - 1;
        for (int j = 1; j < decomp.N_y + 1; ++j) {
            int global_j = decomp.j_start + j - 1;
            for (int k = 1; k < decomp.N_z + 1; ++k) {
                int global_k = decomp.k_start + k - 1;

                if (global_i == 0 || global_i == N - 1 || global_j == 0 || global_j == N - 1 || global_k == 0 ||
                    global_k == N - 1) {
                    u[get_index(i, j, k)] = 0.0;
                } else {
                    double x = global_i * h;
                    double y = global_j * h;
                    double z = global_k * h;
                    u[get_index(i, j, k)] = analytic_solution(x, y, z, T);
                }
            }
        }
    }

    return u;
}