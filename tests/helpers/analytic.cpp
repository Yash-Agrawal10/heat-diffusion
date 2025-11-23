#include "analytic.hpp"

#include <cmath>

double u_analytic(double x, double y, double z, double t) {
    return std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z) * std::exp(-3 * M_PI * M_PI * t);
}