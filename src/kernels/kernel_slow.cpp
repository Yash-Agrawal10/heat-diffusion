#include "kernels/kernels.hpp"

void kernel_slow(std::vector<double>& u, std::vector<double>& u_new, int N, double lambda) {
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            for (int k = 1; k < N - 1; ++k) {
                int idx = i * N * N + j * N + k;
                u_new[idx] =
                    u[idx] + lambda * (u[(i + 1) * N * N + j * N + k] + u[(i - 1) * N * N + j * N + k] +
                                       u[i * N * N + (j + 1) * N + k] + u[i * N * N + (j - 1) * N + k] +
                                       u[i * N * N + j * N + (k + 1)] + u[i * N * N + j * N + (k - 1)] - 6.0 * u[idx]);
            }
        }
    }
}