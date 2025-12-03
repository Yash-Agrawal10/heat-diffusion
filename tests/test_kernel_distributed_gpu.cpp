#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "helpers/analytic.hpp"
#include "helpers/compare.hpp"
#include "solvers/solvers.hpp"
#include "util/decomposition.hpp"
#include "util/problem_spec.hpp"

#include <mpi.h>

TEST_CASE("test distributed_gpu kernel for N=32, T=0.1") {
    // MPI Rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Define constants
    const int N = 32;
    const double T = 0.1;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Create decomposition
    Decomposition decomp = make_decomposition(N, MPI_COMM_WORLD);

    // Get solutions
    auto u_numerical = solver_distributed_gpu_helper(spec, Mode::eval, false, decomp);
    auto u_analytic = analytic_distributed(N, decomp, T);

    // Compare solver solution to analytic solution
    double local_max_error = get_max_error_distributed(u_numerical, u_analytic, decomp.N_x, decomp.N_y, decomp.N_z);
    double global_max_error = 0.0;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        CHECK(global_max_error < epsilon);
    }
}

TEST_CASE("test distributed_gpu kernel for N=64, T=0.05") {
    // MPI Rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Define constants
    const int N = 64;
    const double T = 0.05;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Create decomposition
    Decomposition decomp = make_decomposition(N, MPI_COMM_WORLD);

    // Get solutions
    auto u_numerical = solver_distributed_gpu_helper(spec, Mode::eval, false, decomp);
    auto u_analytic = analytic_distributed(N, decomp, T);

    // Compare solver solution to analytic solution
    double local_max_error = get_max_error_distributed(u_numerical, u_analytic, decomp.N_x, decomp.N_y, decomp.N_z);
    double global_max_error = 0.0;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        CHECK(global_max_error < epsilon);
    }
}

TEST_CASE("test distributed_gpu kernel for N=100, T=0.1") {
    // MPI Rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Define constants
    const int N = 100;
    const double T = 0.1;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Create decomposition
    Decomposition decomp = make_decomposition(N, MPI_COMM_WORLD);

    // Get solutions
    auto u_numerical = solver_distributed_gpu_helper(spec, Mode::eval, false, decomp);
    auto u_analytic = analytic_distributed(N, decomp, T);

    // Compare solver solution to analytic solution
    double local_max_error = get_max_error_distributed(u_numerical, u_analytic, decomp.N_x, decomp.N_y, decomp.N_z);
    double global_max_error = 0.0;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        CHECK(global_max_error < epsilon);
    }
}

TEST_CASE("test distributed_gpu kernel for N=64, T=1.0") {
    // MPI Rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Define constants
    const int N = 64;
    const double T = 1.0;
    ProblemSpec spec{ N, T, analytic_initial_condition };
    const double epsilon = 1e-3;

    // Create decomposition
    Decomposition decomp = make_decomposition(N, MPI_COMM_WORLD);

    // Get solutions
    auto u_numerical = solver_distributed_gpu_helper(spec, Mode::eval, false, decomp);
    auto u_analytic = analytic_distributed(N, decomp, T);

    // Compare solver solution to analytic solution
    double local_max_error = get_max_error_distributed(u_numerical, u_analytic, decomp.N_x, decomp.N_y, decomp.N_z);
    double global_max_error = 0.0;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        CHECK(global_max_error < epsilon);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int res = context.run();
    MPI_Finalize();
    return res;
}