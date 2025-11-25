#include <vector>
#include <cmath>

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