#pragma once

#include "util/problem_spec.hpp"

std::vector<double> solver_slow(const ProblemSpec& spec, Mode mode, bool verbose = false);
std::vector<double> solver_fast(const ProblemSpec& spec, Mode mode, bool verbose = false);