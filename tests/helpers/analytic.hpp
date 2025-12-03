#pragma once

#include "util/decomposition.hpp"

#include <cmath>
#include <vector>

inline double analytic_solution(double x, double y, double z, double t) {
    return std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z) * std::exp(-3 * M_PI * M_PI * t);
}

inline double analytic_initial_condition(double x, double y, double z) { return analytic_solution(x, y, z, 0.0); }

std::vector<double> analytic_unit_cube(int N, double T);

std::vector<double> analytic_distributed(int N, Decomposition& decomp, double T);