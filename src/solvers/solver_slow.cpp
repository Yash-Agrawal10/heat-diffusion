#include "solvers/solvers.hpp"

#include "kernels/kernels.hpp"
#include "util/constants.hpp"
#include "util/initial_condition.hpp"
#include "util/output.hpp"

#include <chrono>
#include <iostream>

std::vector<double> solver_slow(const ProblemSpec& spec, Mode mode, bool verbose) {
    using Clock = std::chrono::high_resolution_clock;

    // Define constants
    Constants consts = compute_constants(spec);

    // Create initial condition grid
    auto u = initial_condition_unit_cube(spec.N, spec.initial_condition);
    std::vector<double> u_new(spec.N * spec.N * spec.N, 0.0);

    // Main time-stepping loop
    auto start = Clock::now();
    const int num_frames = 60;
    const int output_interval = std::max(1, consts.n_steps / num_frames);

    if (mode == Mode::output) {
        dump_state(u, spec.N, 0);
    }
    if (verbose) {
        std::cout << "Starting slow solver with " << consts.n_steps << " time steps...\n";
    }

    for (int step = 0; step < consts.n_steps; ++step) {
        kernel_slow(u, u_new, spec.N, consts.lambda);
        std::swap(u, u_new);

        if (mode == Mode::output && (step + 1) % output_interval == 0 && (step + 1) != consts.n_steps) {
            dump_state(u, spec.N, step + 1);
        }
        if (verbose && (step + 1) % (consts.n_steps / 10) == 0) {
            std::cout << "Completed " << (step + 1) << " / " << consts.n_steps << " steps...\n";
        }
    }

    if (mode == Mode::output) {
        dump_state(u, spec.N, consts.n_steps);
    }
    if (verbose) {
        std::cout << "Finished slow solver.\n" << std::endl;
    }
    auto end = Clock::now();
    if (mode == Mode::profile) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n" << std::endl;
    }

    return u;
}