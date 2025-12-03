#include "solvers/solvers.hpp"
#include "util/problem_spec.hpp"

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

void print_usage() {
    std::cout << "Usage: ./heat_diffusion -N <num_grid_points> -T <total_time> --mode <eval|profile|output> --kernel "
                 "<cpu|shared_gpu|distributed_gpu>\n";
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int local_size = 0;
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);
    int nodes = size / local_size;

    // Get number of available HIP devices
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        std::cerr << "Failed to get HIP device count: " << hipGetErrorString(err) << std::endl;
        device_count = -1;
    }

    // Parse command line arguments
    int N = -1;
    double T = -1;
    std::string kernel_str = "";
    std::string mode_str = "";
    bool verbose = false;

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
        } else if (arg == "--verbose") {
            verbose = true;
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

    Kernel kernel = Kernel::cpu;
    if (kernel_str == "cpu") {
        kernel = Kernel::cpu;
    } else if (kernel_str == "shared_gpu") {
        kernel = Kernel::shared_gpu;
    } else if (kernel_str == "distributed_gpu") {
        kernel = Kernel::distributed_gpu;
    } else {
        std::cerr << "Error: Invalid kernel specified." << std::endl;
        print_usage();
        return 1;
    }
    if (kernel != Kernel::distributed_gpu && size > 1) {
        if (rank == 0) {
            std::cerr << "Only the distributed_gpu kernel supports multiple MPI ranks." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    if (rank == 0) {
        std::cout << "Heat Diffusion Simulation\n";
        std::cout << "Grid points (N): " << N << "\n";
        std::cout << "Total time (T): " << T << "\n";
        std::cout << "Mode: " << mode_str << "\n";
        std::cout << "Kernel: " << kernel_str << "\n";
        std::cout << "MPI Ranks: " << size << "\n";
        std::cout << "Nodes: " << nodes << "\n";
        std::cout << "Num devices (on node containing rank 0): " << device_count << "\n";
        std::cout << std::endl;
    }

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
    if (kernel == Kernel::cpu) {
        u = solver_cpu(spec, mode, verbose);
    } else if (kernel == Kernel::shared_gpu) {
        u = solver_shared_gpu(spec, mode, verbose);
    } else if (kernel == Kernel::distributed_gpu) {
        u = solver_distributed_gpu(spec, mode, verbose);
    } else {
        if (rank == 0) {
            std::cerr << "Error: Unknown kernel." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Return success
    MPI_Finalize();
    return 0;
}