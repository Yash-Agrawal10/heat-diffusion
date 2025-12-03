#include "solvers/solvers.hpp"

#include "kernels/kernels.hpp"
#include "util/constants.hpp"
#include "util/helpers.hpp"

#include <chrono>
#include <iostream>

std::vector<double> solver_slow(const ProblemSpec& spec, Mode mode) {
    using Clock = std::chrono::high_resolution_clock;

    // Define constants
    Constants consts = compute_constants(spec);

    // Create initial condition grid
    auto u = initial_condition_unit_cube(spec.N, spec.initial_condition);
    std::vector<double> u_new(spec.N * spec.N * spec.N, 0.0);

    // Select mode
    if (mode == Mode::profile) {
        auto start = Clock::now();
        for (int i = 0; i < consts.n_steps; ++i) {
            kernel_slow(u, u_new, spec.N, consts.lambda);
            std::swap(u, u_new);
        }
        auto end = Clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    } else if (mode == Mode::output) {
        const int num_frames = 60;
        const int output_interval = std::max(1, consts.n_steps / num_frames);
        dump_state(u, spec.N, 0);
        for (int i = 0; i < consts.n_steps; ++i) {
            kernel_slow(u, u_new, spec.N, consts.lambda);
            std::swap(u, u_new);
            if ((i + 1) % output_interval == 0 && (i + 1) != consts.n_steps) {
                dump_state(u, spec.N, i + 1);
            }
        }
        dump_state(u, spec.N, consts.n_steps);
    } else if (mode == Mode::eval) {
        for (int step = 0; step < consts.n_steps; ++step) {
            kernel_slow(u, u_new, spec.N, consts.lambda);
            std::swap(u, u_new);
        }
    }

    return u;
}