#include "solvers/solvers.hpp"
#include "util/problem_spec.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

void print_usage() {
    std::cout << "Usage: ./heat_diffusion -N <num_grid_points> -T <total_time> --mode <eval|profile|output> --kernel "
                 "<slow|fast>\n";
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Temporarily only run rank 0
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    // Version number for printing
    int version = 2;

    // Parse command line arguments
    int N = -1;
    double T = -1;
    std::string kernel_str = "";
    std::string mode_str = "";

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "-T" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_str = argv[++i];
        } else if (arg == "--kernel" && i + 1 < argc) {
            kernel_str = argv[++i];
        }
    }

    Mode mode = Mode::eval;
    if (mode_str == "profile") {
        mode = Mode::profile;
    } else if (mode_str == "output") {
        mode = Mode::output;
    } else if (mode_str == "eval") {
        mode = Mode::eval;
    } else {
        std::cerr << "Error: Invalid mode specified." << std::endl;
        print_usage();
        return 1;
    }

    Kernel kernel = Kernel::fast;
    if (kernel_str == "slow") {
        kernel = Kernel::slow;
    } else if (kernel_str == "fast") {
        kernel = Kernel::fast;
    } else {
        std::cerr << "Error: Invalid kernel specified." << std::endl;
        print_usage();
        return 1;
    }

    if (N <= 0) {
        std::cerr << "Error: Number of grid points N must be positive." << std::endl;
        print_usage();
        return 1;
    }

    if (T < 0) {
        std::cerr << "Error: Total simulation time T must be non-negative." << std::endl;
        print_usage();
        return 1;
    }

    // Print problem details
    std::cout << "Heat Diffusion Simulation\n";
    std::cout << "Version: " << version << "\n";
    std::cout << "Grid points (N): " << N << "\n";
    std::cout << "Total time (T): " << T << "\n";
    std::cout << "Mode: " << mode_str << "\n";
    std::cout << "Kernel: " << kernel_str << "\n";
    std::cout << std::endl;

    // Define problem
    auto initial_condition = [](double x, double y, double z) {
        if (std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5)) < 0.25) {
            return 1.0;
        } else {
            return 0.0;
        }
    };
    ProblemSpec spec{ N, T, initial_condition };

    std::vector<double> u;
    // Call correct solver
    if (kernel == Kernel::slow) {
        u = solver_slow(spec, mode);
    } else {
        u = solver_fast(spec, mode);
    }

    // Return success
    MPI_Finalize();
    return 0;
}