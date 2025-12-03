#pragma once

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

inline std::vector<double> initial_condition_distributed(int N, int N_x, int N_y, int N_z, int px, int py, int pz,
                                                         int Px, int Py, int Pz, IC_Func ic_func) {
    std::vector<double> u((N_x + 2) * (N_y + 2) * (N_z + 2), 0.0);
    const double h = 1.0 / (N - 1);

    auto compute_start = [](int N, int P, int p) {
        int base = N / P;
        int remainder = N % P;
        return p * base + std::min(p, remainder);
    };
    
    int i_start = compute_start(N, Px, px);
    int j_start = compute_start(N, Py, py);
    int k_start = compute_start(N, Pz, pz);

    auto get_index = [=](int i, int j, int k) {
        return i * (N_y + 2) * (N_z + 2) + j * (N_z + 2) + k;
    };

    for (int i = 1; i < N_x + 1; ++i) {
        int global_i = i_start + i - 1;
        for (int j = 1; j < N_y + 1; ++j) {
            int global_j = j_start + j - 1;
            for (int k = 1; k < N_z + 1; ++k) {
                int global_k = k_start + k - 1;

                if (global_i == 0 || global_i == N - 1 ||
                    global_j == 0 || global_j == N - 1 ||
                    global_k == 0 || global_k == N - 1) {
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