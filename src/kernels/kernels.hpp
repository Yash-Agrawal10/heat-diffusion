#pragma once

#include <vector>

void kernel_slow(std::vector<double>& u, std::vector<double>& u_new, int N, double lambda);
void kernel_fast(std::vector<double>& u, std::vector<double>& u_new, int N, double lambda);