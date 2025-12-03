#pragma once

#include "util/problem_spec.hpp"

std::vector<double> solver_cpu(const ProblemSpec& spec, Mode mode, bool verbose = false);
std::vector<double> solver_shared_gpu(const ProblemSpec& spec, Mode mode, bool verbose = false);
std::vector<double> solver_distributed_gpu(const ProblemSpec& spec, Mode mode, bool verbose = false);