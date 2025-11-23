#include <cmath>
#include <iostream>
#include <vector>

void heat_diffusion_3d(int N, double T, std::vector<std::vector<std::vector<double>>>& u) {
    // Define constants
    const double h = 1.0 / (N - 1);
    const double dt = 0.5 * h * h / 6.0;

    // Initialize update grid
    std::vector<std::vector<std::vector<double>>> u_new = u;

    // Time-stepping loop
    for (double t = 0; t < T; t += dt) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                for (int k = 1; k < N - 1; ++k) {
                    u_new[i][j][k] = u[i][j][k] + dt * ((u[i + 1][j][k] - 2 * u[i][j][k] + u[i - 1][j][k]) / (h * h) +
                                                        (u[i][j + 1][k] - 2 * u[i][j][k] + u[i][j - 1][k]) / (h * h) +
                                                        (u[i][j][k + 1] - 2 * u[i][j][k] + u[i][j][k - 1]) / (h * h));
                }
            }
        }
        u.swap(u_new);
    }
}
