#include <cmath>
#include <vector>

inline double get_max_error(const std::vector<double>& u_numerical, const std::vector<double>& u_analytic, int N) {
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int index = i * N * N + j * N + k;
                double error = std::abs(u_numerical[index] - u_analytic[index]);
                max_err = std::max(max_err, error);
            }
        }
    }
    return max_err;
}

inline double get_max_error_distributed(const std::vector<double>& u_numerical, const std::vector<double>& u_analytic,
                                       int N_x, int N_y, int N_z) {
    double max_err = 0.0;
    for (int i = 1; i <= N_x; ++i) {
        for (int j = 1; j <= N_y; ++j) {
            for (int k = 1; k <= N_z; ++k) {
                int index = i * (N_y + 2) * (N_z + 2) + j * (N_z + 2) + k;
                double error = std::abs(u_numerical[index] - u_analytic[index]);
                max_err = std::max(max_err, error);
            }
        }
    }
    return max_err;
}