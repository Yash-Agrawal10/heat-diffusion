#pragma once

#include <cmath>
#include <vector>

using IC_Func = double (*)(double, double, double);

struct ProblemSpec {
    int N;                     // Number of grid points in each dimension
    double T;                  // Total simulation time
    IC_Func initial_condition; // Function pointer for initial condition
};

struct Constants {
    int N;
    double T;
    double h;
    double lambda;
    double dt;
    int n_steps;
};

inline std::vector<double> initial_condition_unit_cube(int N, IC_Func ic_func) {
    std::vector<double> u(N * N * N);
    const double h = 1.0 / (N - 1);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * h;
                double y = j * h;
                double z = k * h;
                u[i * N * N + j * N + k] = ic_func(x, y, z);
            }
        }
    }

    return u;
}

inline Constants compute_constants(const ProblemSpec& spec) {
    const int N = spec.N;
    const double T = spec.T;
    const double h = 1.0 / (N - 1);

    const double lambda = 1.0 / 12.0;
    const double dt = lambda * h * h;
    const int n_steps = std::round(T / dt);
    const double dt_eff = T / n_steps;
    const double lambda_eff = dt_eff / (h * h);

    return Constants{ N, T, h, lambda_eff, dt_eff, n_steps };
}