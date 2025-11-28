cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER /opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/bin/mpicxx)
project(heat-diffusion
    VERSION 1.0
    DESCRIPTION "HPC 3d heat diffusion simulation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

add_subdirectory(src)

include(CTest)
add_subdirectory(tests)