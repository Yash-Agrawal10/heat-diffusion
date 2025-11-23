#include "analytic.hpp"

#include <cmath>

double u_analytic(double x, double y, double z, double t) {
    return std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z) * std::exp(-3 * M_PI * M_PI * t);
}

double u_analytic_unit_cube(double x_index, double y_index, double z_index, double t, double h) {
    return u_analytic(x_index * h, y_index * h, z_index * h, t);
}