#include "kernel_slow.hpp"

#include "problem_spec.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

std::vector<double> heat_diffuision_kernel_slow(const Constants& consts, std::vector<double>& u,
                                                std::vector<double>& u_new, bool verbose) {
    (void)verbose;  // Suppress unused variable warning
    const int N = consts.N;

#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    for (int n = 0; n < consts.n_steps; ++n) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                for (int k = 1; k < N - 1; ++k) {
                    int idx = i * N * N + j * N + k;
                    u_new[idx] =
                        u[idx] + consts.lambda *
                                     (u[(i + 1) * N * N + j * N + k] + u[(i - 1) * N * N + j * N + k] +
                                      u[i * N * N + (j + 1) * N + k] + u[i * N * N + (j - 1) * N + k] +
                                      u[i * N * N + j * N + (k + 1)] + u[i * N * N + j * N + (k - 1)] - 6.0 * u[idx]);
                }
            }
        }
        u.swap(u_new);
    }
#ifdef PROFILE
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "# Timing\n"
              << "kernel_compute=" << elapsed.count() << " s\n";
#endif
    return u;
}