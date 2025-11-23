#include "solver.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // Define constants
    const int N = 10;
    const double h = 1.0 / (N - 1);
    const double T = 0.1;

    // Initialize temperature grid
    std::vector<std::vector<std::vector<double>>> u(N,
                                                    std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0)));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
                    u[i][j][k] = 0.0;
                } else {
                    u[i][j][k] = std::sin(M_PI * i * h) * std::sin(M_PI * j * h) * std::sin(M_PI * k * h);
                }
            }
        }
    }

    // Perform heat diffusion
    heat_diffusion_3d(N, T, u);

    // Print results at the center slice
    std::cout << "Temperature distribution at center slice (k = " << N / 2 << "):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << u[i][j][N / 2] << " ";
        }
        std::cout << "\n";
    }
}