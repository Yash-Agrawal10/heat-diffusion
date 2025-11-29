#include "kernel_fast.hpp"
#include "kernel_slow.hpp"
#include "problem_spec.hpp"
#include "solver.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int N = -1;
    double T = -1;
    std::string mode = "";
    bool verbose = false;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "-T" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }

    if (N < 0 || T < 0 || (mode != "slow" && mode != "fast")) {
        std::cerr << "Usage: " << argv[0] << " -N <int> -T <double> --mode <slow|fast> [--verbose]\n";
        return 1;
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

    // Perform heat diffusion
    auto u = heat_diffusion_solver(spec, mode == "slow" ? heat_diffusion_kernel_slow : heat_diffusion_kernel_fast, verbose);

    // Temporarily output something for non-profile runs
#ifndef PROFILE
    std::cout << "Final value at center: " << u[(N / 2) * N * N + (N / 2) * N + (N / 2)] << "\n";
#endif

    // Return success
    return 0;
}