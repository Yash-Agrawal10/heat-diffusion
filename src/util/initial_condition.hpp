#pragma once

#include "util/decomposition.hpp"
#include "util/problem_spec.hpp"

#include <vector>

inline std::vector<double> initial_condition_unit_cube(int N, IC_Func ic_func) {
    std::vector<double> u(N * N * N, 0.0);
    const double h = 1.0 / (N - 1);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
                    u[i * N * N + j * N + k] = 0.0;
                } else {
                    double x = i * h;
                    double y = j * h;
                    double z = k * h;
                    u[i * N * N + j * N + k] = ic_func(x, y, z);
                }
            }
        }
    }

    return u;
}

inline std::vector<double> initial_condition_distributed(int N, Decomposition decomp, IC_Func ic_func) {
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
                    u[get_index(i, j, k)] = ic_func(x, y, z);
                }
            }
        }
    }

    return u;
}