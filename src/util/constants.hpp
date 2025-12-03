#pragma once

#include "problem_spec.hpp"

#include <cmath>

struct Constants {
    int N;
    double T;
    double h;
    double lambda;
    double dt;
    int n_steps;
};

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
