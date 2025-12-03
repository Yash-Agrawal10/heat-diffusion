#pragma once

#include <fstream>
#include <vector>

inline void dump_state(std::vector<double>& u, int N, int step) {
    const double h = 1.0 / (N - 1);

    std::ofstream file;
    file.open("heat_diffusion_" + std::to_string(step) + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file.");
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "Heat diffusion data\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << N << " " << N << " " << N << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << h << " " << h << " " << h << "\n";
    file << "POINT_DATA " << N * N * N << "\n";
    file << "SCALARS temperature double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < N * N * N; ++i) {
        file << u[i] << "\n";
    }
    file.close();
}