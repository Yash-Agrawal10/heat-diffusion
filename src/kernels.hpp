#pragma once

#include "problem_spec.hpp"

#include <vector>

std::vector<double> heat_diffusion_kernel_slow(const Constants& consts, std::vector<double>& u,
                                               std::vector<double>& u_new, bool verbose = false);

std::vector<double> heat_diffusion_kernel_fast(const Constants& consts, std::vector<double>& u,
                                               std::vector<double>& u_new, bool verbose = false);