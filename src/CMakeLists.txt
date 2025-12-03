add_library(kernels STATIC kernels/kernel_slow.cpp kernels/kernel_fast.cpp)
target_include_directories(kernels PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/opt/rocm-7.1.0/include"
    "/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include"
)

add_library(solvers STATIC solvers/solver_slow.cpp solvers/solver_fast.cpp)
target_link_libraries(solvers PRIVATE kernels)
target_include_directories(solvers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(heat_diffusion main.cpp)
target_link_libraries(heat_diffusion PRIVATE solvers)
target_include_directories(heat_diffusion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})