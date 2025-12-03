#pragma once

#include <mpi.h>

struct Decomposition {
    MPI_Comm cart_comm;
    int rank;
    int dims[3];
    int coords[3];
    int N_x, N_y, N_z;
    int i_start, j_start, k_start;
};

inline Decomposition make_decomposition(int N, MPI_Comm world) {
    int rank, size;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    int dims[3] = { 0, 0, 0 };
    MPI_Dims_create(size, 3, dims);

    int periods[3] = { 0, 0, 0 };
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    auto compute_start = [](int N, int P, int p) {
        int base = N / P;
        int remainder = N % P;
        return p * base + std::min(p, remainder);
    };

    int i_start = compute_start(N, dims[0], coords[0]);
    int j_start = compute_start(N, dims[1], coords[1]);
    int k_start = compute_start(N, dims[2], coords[2]);

    return Decomposition{ .cart_comm = cart_comm,
                          .rank = rank,
                          .dims = { dims[0], dims[1], dims[2] },
                          .coords = { coords[0], coords[1], coords[2] },
                          .N_x = (coords[0] < N % dims[0]) ? (N / dims[0] + 1) : (N / dims[0]),
                          .N_y = (coords[1] < N % dims[1]) ? (N / dims[1] + 1) : (N / dims[1]),
                          .N_z = (coords[2] < N % dims[2]) ? (N / dims[2] + 1) : (N / dims[2]),
                          .i_start = i_start,
                          .j_start = j_start,
                          .k_start = k_start };
}