#pragma once

#include <cmath>
#include <fstream>
#include <vector>

using IC_Func = double (*)(double, double, double);

struct ProblemSpec {
    int N;                     // Number of grid points in each dimension
    double T;                  // Total simulation time
    IC_Func initial_condition; // Function pointer for initial condition
};

enum class Mode { profile, output, eval };

enum class Kernel { slow, fast };

inline std::vector<double> initial_condition_unit_cube(int N, IC_Func ic_func) {
    std::vector<double> u(N * N * N);
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
