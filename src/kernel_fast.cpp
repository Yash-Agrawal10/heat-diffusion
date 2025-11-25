#include "kernel_slow.hpp"

#include "problem_spec.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

std::vector<double> heat_diffusion_kernel_fast(const Constants& consts, std::vector<double>& u,
                                                std::vector<double>& u_new, bool verbose) {
    const int N = consts.N;
#ifdef PROFILE
    (void)verbose;  // Suppress unused variable warning
#endif

#ifndef PROFILE
    const int total_outputs = 60;
    const int output_interval = std::max(1, consts.n_steps / total_outputs);
#endif

#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif

#ifndef PROFILE
    if (verbose) {
        dump_state(u, N, 0);
    }
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

#ifndef PROFILE
        int step = n + 1;
        if (verbose && step != consts.n_steps && step % output_interval == 0) {
            dump_state(u, N, step);
        }
#endif
    }

#ifdef PROFILE
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "# Timing\n"
              << "kernel_compute=" << elapsed.count() << " s\n";
#endif

#ifndef PROFILE
    if (verbose) {
        dump_state(u, N, consts.n_steps);
    }
#endif

    return u;
}