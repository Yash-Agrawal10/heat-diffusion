cmake_minimum_required(VERSION 3.16)
project(heat-diffusion
    VERSION 1.0
    DESCRIPTION "HPC 3d heat diffusion simulation"
    LANGUAGES CXX HIP
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

include(CTest)
add_subdirectory(tests)