#include "solvers/solvers.hpp"

#include "kernels/kernels.hpp"
#include "util/constants.hpp"
#include "util/initial_condition.hpp"
#include "util/output.hpp"

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <chrono>
#include <iostream>

void check_hip_error(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

std::vector<double> solver_distributed_gpu(const ProblemSpec& spec, Mode mode, bool verbose) {
    using Clock = std::chrono::high_resolution_clock;

    // Temporarily disable output mode
    if (mode == Mode::output) {
        std::cout << "Output mode is not yet supported in the distributed GPU solver. Switching to eval mode.\n";
        mode = Mode::eval;
    }

    // Create MPI grid
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[3] = { 0, 0, 0 };
    MPI_Dims_create(size, 3, dims);
    int periods[3] = { 0, 0, 0 };
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    int px = coords[0];
    int py = coords[1];
    int pz = coords[2];

    // Bind ranks to GPUs
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    int device_id = rank % device_count;
    hipSetDevice(device_id);

    // Define constants
    Constants consts = compute_constants(spec);
    const int N = consts.N;
    const int N_x = (px < N % dims[0]) ? (N / dims[0] + 1) : (N / dims[0]);
    const int N_y = (py < N % dims[1]) ? (N / dims[1] + 1) : (N / dims[1]);
    const int N_z = (pz < N % dims[2]) ? (N / dims[2] + 1) : (N / dims[2]);
    const int N_x_total = N_x + 2;
    const int N_y_total = N_y + 2;
    const int N_z_total = N_z + 2;

    // Define GPU thread layout
    const dim3 kernel_block_size(32, 4, 1);
    const dim3 kernel_grid_size((N_z_total + kernel_block_size.x - 1) / kernel_block_size.x,
                                (N_y_total + kernel_block_size.y - 1) / kernel_block_size.y,
                                (N_x_total + kernel_block_size.z - 1) / kernel_block_size.z);

    const dim3 pack_x_block_size(32, 4);
    const dim3 pack_x_grid_size((N_z_total + pack_x_block_size.x - 1) / pack_x_block_size.x,
                                (N_y_total + pack_x_block_size.y - 1) / pack_x_block_size.y);

    const dim3 pack_y_block_size(32, 4);
    const dim3 pack_y_grid_size((N_z_total + pack_y_block_size.x - 1) / pack_y_block_size.x,
                                (N_x_total + pack_y_block_size.y - 1) / pack_y_block_size.y);

    const dim3 pack_z_block_size(32, 4);
    const dim3 pack_z_grid_size((N_y_total + pack_z_block_size.x - 1) / pack_z_block_size.x,
                                (N_x_total + pack_z_block_size.y - 1) / pack_z_block_size.y);

    // Set up MPI halo exchanges
    int nbr_xm, nbr_xp;
    int nbr_ym, nbr_yp;
    int nbr_zm, nbr_zp;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_xm, &nbr_xp);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_ym, &nbr_yp);
    MPI_Cart_shift(cart_comm, 2, 1, &nbr_zm, &nbr_zp);
    MPI_Request requests[12];

    // Allocate and initialize host memory
    auto u =
        initial_condition_distributed(N, N_x, N_y, N_z, px, py, pz, dims[0], dims[1], dims[2], spec.initial_condition);
    std::vector<double> u_new(N_x_total * N_y_total * N_z_total, 0.0);

    // Allocate and initialize device memory
    if (verbose && rank == 0) {
        std::cout << "Allocating and initializing device memory...\n";
    }
    auto device_alloc_start = Clock::now();
    double *d_u, *d_u_new;
    double *d_send_xm, *d_send_xp;
    double *d_send_ym, *d_send_yp;
    double *d_send_zm, *d_send_zp;
    double *d_recv_xm, *d_recv_xp;
    double *d_recv_ym, *d_recv_yp;
    double *d_recv_zm, *d_recv_zp;
    hipMalloc(&d_u, u.size() * sizeof(double));
    hipMalloc(&d_u_new, u_new.size() * sizeof(double));
    hipMalloc(&d_send_xm, N_y * N_z * sizeof(double));
    hipMalloc(&d_send_xp, N_y * N_z * sizeof(double));
    hipMalloc(&d_send_ym, N_x * N_z * sizeof(double));
    hipMalloc(&d_send_yp, N_x * N_z * sizeof(double));
    hipMalloc(&d_send_zm, N_x * N_y * sizeof(double));
    hipMalloc(&d_send_zp, N_x * N_y * sizeof(double));
    hipMalloc(&d_recv_xm, N_y * N_z * sizeof(double));
    hipMalloc(&d_recv_xp, N_y * N_z * sizeof(double));
    hipMalloc(&d_recv_ym, N_x * N_z * sizeof(double));
    hipMalloc(&d_recv_yp, N_x * N_z * sizeof(double));
    hipMalloc(&d_recv_zm, N_x * N_y * sizeof(double));
    hipMalloc(&d_recv_zp, N_x * N_y * sizeof(double));
    hipMemcpy(d_u, u.data(), u.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_u_new, u_new.data(), u_new.size() * sizeof(double), hipMemcpyHostToDevice);
    auto device_alloc_end = Clock::now();
    if (verbose && rank == 0) {
        std::cout << "Device memory allocation and initialization complete.\n";
    }
    if (mode == Mode::profile && rank == 0) {
        std::chrono::duration<double> alloc_elapsed = device_alloc_end - device_alloc_start;
        std::cout << "Device memory allocation and initialization time: " << alloc_elapsed.count() << " seconds\n";
    }

    // Main time-stepping loop
    if (verbose && rank == 0) {
        std::cout << "Starting main time-stepping loop...\n";
    }

    auto start = Clock::now();
    const int num_frames = 60;
    const int output_interval = std::max(1, consts.n_steps / num_frames);

    for (int step = 0; step < consts.n_steps; ++step) {
        // Exchange halos
        pack_x<<<pack_x_grid_size, pack_x_block_size>>>(d_u, d_send_xm, N_x, N_y, N_z, FaceSide::minus);
        pack_x<<<pack_x_grid_size, pack_x_block_size>>>(d_u, d_send_xp, N_x, N_y, N_z, FaceSide::plus);
        pack_y<<<pack_y_grid_size, pack_y_block_size>>>(d_u, d_send_ym, N_x, N_y, N_z, FaceSide::minus);
        pack_y<<<pack_y_grid_size, pack_y_block_size>>>(d_u, d_send_yp, N_x, N_y, N_z, FaceSide::plus);
        pack_z<<<pack_z_grid_size, pack_z_block_size>>>(d_u, d_send_zm, N_x, N_y, N_z, FaceSide::minus);
        pack_z<<<pack_z_grid_size, pack_z_block_size>>>(d_u, d_send_zp, N_x, N_y, N_z, FaceSide::plus);
        check_hip_error(hipDeviceSynchronize());

        int r = 0;

        MPI_Irecv(d_recv_xm, N_y * N_z, MPI_DOUBLE, nbr_xm, 0, cart_comm, &requests[r++]);
        MPI_Irecv(d_recv_xp, N_y * N_z, MPI_DOUBLE, nbr_xp, 1, cart_comm, &requests[r++]);
        MPI_Irecv(d_recv_ym, N_x * N_z, MPI_DOUBLE, nbr_ym, 2, cart_comm, &requests[r++]);
        MPI_Irecv(d_recv_yp, N_x * N_z, MPI_DOUBLE, nbr_yp, 3, cart_comm, &requests[r++]);
        MPI_Irecv(d_recv_zm, N_x * N_y, MPI_DOUBLE, nbr_zm, 4, cart_comm, &requests[r++]);
        MPI_Irecv(d_recv_zp, N_x * N_y, MPI_DOUBLE, nbr_zp, 5, cart_comm, &requests[r++]);

        MPI_Isend(d_send_xp, N_y * N_z, MPI_DOUBLE, nbr_xp, 0, cart_comm, &requests[r++]);
        MPI_Isend(d_send_xm, N_y * N_z, MPI_DOUBLE, nbr_xm, 1, cart_comm, &requests[r++]);
        MPI_Isend(d_send_yp, N_x * N_z, MPI_DOUBLE, nbr_yp, 2, cart_comm, &requests[r++]);
        MPI_Isend(d_send_ym, N_x * N_z, MPI_DOUBLE, nbr_ym, 3, cart_comm, &requests[r++]);
        MPI_Isend(d_send_zp, N_x * N_y, MPI_DOUBLE, nbr_zp, 4, cart_comm, &requests[r++]);
        MPI_Isend(d_send_zm, N_x * N_y, MPI_DOUBLE, nbr_zm, 5, cart_comm, &requests[r++]);

        MPI_Waitall(r, requests, MPI_STATUSES_IGNORE);

        unpack_x<<<pack_x_grid_size, pack_x_block_size>>>(d_recv_xm, d_u, N_x, N_y, N_z, FaceSide::minus);
        unpack_x<<<pack_x_grid_size, pack_x_block_size>>>(d_recv_xp, d_u, N_x, N_y, N_z, FaceSide::plus);
        unpack_y<<<pack_y_grid_size, pack_y_block_size>>>(d_recv_ym, d_u, N_x, N_y, N_z, FaceSide::minus);
        unpack_y<<<pack_y_grid_size, pack_y_block_size>>>(d_recv_yp, d_u, N_x, N_y, N_z, FaceSide::plus);
        unpack_z<<<pack_z_grid_size, pack_z_block_size>>>(d_recv_zm, d_u, N_x, N_y, N_z, FaceSide::minus);
        unpack_z<<<pack_z_grid_size, pack_z_block_size>>>(d_recv_zp, d_u, N_x, N_y, N_z, FaceSide::plus);

        // Execute kernel
        kernel_distributed_gpu<<<kernel_grid_size, kernel_block_size>>>(d_u, d_u_new, N_x, N_y, N_z, consts.lambda);

        // Swap pointers
        std::swap(d_u, d_u_new);

        if (verbose && rank == 0 && (step + 1) % (consts.n_steps / 100) == 0) {
            std::cout << "Completed step " << (step + 1) << " / " << consts.n_steps << "\n";
        }
    }

    check_hip_error(hipDeviceSynchronize());
    if (verbose && rank == 0) {
        std::cout << "Main time-stepping loop complete.\n";
    }
    auto end = Clock::now();
    if (mode == Mode::profile && rank == 0) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    }

    return u;
}