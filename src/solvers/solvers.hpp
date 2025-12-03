#pragma once

#include "util/decomposition.hpp"
#include "util/problem_spec.hpp"

#include <vector>

std::vector<double> solver_cpu(const ProblemSpec& spec, Mode mode, bool verbose = false);
std::vector<double> solver_shared_gpu(const ProblemSpec& spec, Mode mode, bool verbose = false);
std::vector<double> solver_distributed_gpu(const ProblemSpec& spec, Mode mode, bool verbose = false);
std::vector<double> solver_distributed_gpu_helper(const ProblemSpec& spec, Mode mode, bool verbose,
                                                  Decomposition& decomp);