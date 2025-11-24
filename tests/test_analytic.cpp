#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "helpers/analytic.hpp"

TEST_CASE("test analytic at x boundary") {
    double T = 0.1;
    double y = 0.5;
    double z = 0.5;

    // u_x_t
    double u_0_0 = analytic_initial_condition(0.0, y, z);
    double u_1_0 = analytic_initial_condition(1.0, y, z);
    double u_0_1 = analytic_solution(0.0, y, z, T);
    double u_1_1 = analytic_solution(1.0, y, z, T);

    CHECK(u_0_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_1_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_0_1 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_1_1 == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("test analytic at y boundary") {
    double T = 0.1;
    double x = 0.5;
    double z = 0.5;

    // u_y_t
    double u_0_0 = analytic_initial_condition(x, 0.0, z);
    double u_1_0 = analytic_initial_condition(x, 1.0, z);
    double u_0_1 = analytic_solution(x, 0.0, z, T);
    double u_1_1 = analytic_solution(x, 1.0, z, T);

    CHECK(u_0_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_1_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_0_1 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_1_1 == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("test analytic at z boundary") {
    double T = 0.1;
    double x = 0.5;
    double y = 0.5;

    // u_z_t
    double u_0_0 = analytic_initial_condition(x, y, 0.0);
    double u_1_0 = analytic_initial_condition(x, y, 1.0);
    double u_0_1 = analytic_solution(x, y, 0.0, T);
    double u_1_1 = analytic_solution(x, y, 1.0, T);

    CHECK(u_0_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_1_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_0_1 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_1_1 == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("test analytic at center") {
    double T = 0.1;
    double x = 0.5;
    double y = 0.5;
    double z = 0.5;

    double u_at_center = analytic_solution(x, y, z, T);
    double expected_value =
        std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z) * std::exp(-3 * M_PI * M_PI * T);

    CHECK(u_at_center == doctest::Approx(expected_value).epsilon(1e-12));
}

TEST_CASE("test analytic unit cube") {
    int N = 10;
    double T = 0.1;

    auto u = analytic_unit_cube(N, T);

    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = static_cast<double>(i) / (N - 1);
                double y = static_cast<double>(j) / (N - 1);
                double z = static_cast<double>(k) / (N - 1);
                double expected_value = analytic_solution(x, y, z, T);
                double computed_value = u[i * N * N + j * N + k];
                double error = std::fabs(computed_value - expected_value);
                max_error = std::max(max_error, error);
            }
        }
    }

    CHECK(max_error < 1e-2);
}