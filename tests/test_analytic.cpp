#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "helpers/analytic.hpp"

TEST_CASE("test analytic at x boundary") {
    double t = 0.1;
    double y = 0.5;
    double z = 0.5;

    double u_at_0 = u_analytic(0.0, y, z, t);
    double u_at_1 = u_analytic(1.0, y, z, t);

    CHECK(u_at_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_at_1 == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("test analytic at y boundary") {
    double t = 0.1;
    double x = 0.5;
    double z = 0.5;

    double u_at_0 = u_analytic(x, 0.0, z, t);
    double u_at_1 = u_analytic(x, 1.0, z, t);

    CHECK(u_at_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_at_1 == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("test analytic at z boundary") {
    double t = 0.1;
    double x = 0.5;
    double y = 0.5;

    double u_at_0 = u_analytic(x, y, 0.0, t);
    double u_at_1 = u_analytic(x, y, 1.0, t);

    CHECK(u_at_0 == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(u_at_1 == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("test analytic at center") {
    double t = 0.1;
    double x = 0.5;
    double y = 0.5;
    double z = 0.5;

    double u_at_center = u_analytic(x, y, z, t);
    double expected_value = std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z) * std::exp(-3 * M_PI * M_PI * t);

    CHECK(u_at_center == doctest::Approx(expected_value).epsilon(1e-12));
}