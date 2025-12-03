include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_library(test_helpers 
    helpers/analytic.cpp
)
target_include_directories(test_helpers 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

add_executable(test_analytic test_analytic.cpp)
target_link_libraries(test_analytic PRIVATE
    test_helpers
    doctest::doctest
)
add_test(NAME test_analytic COMMAND srun ./test_analytic)

add_executable(test_kernel_cpu test_kernel_cpu.cpp)
target_link_libraries(test_kernel_cpu PRIVATE
    test_helpers
    solvers
    doctest::doctest
)
target_include_directories(test_kernel_cpu PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME test_kernel_cpu COMMAND srun -n 1 ./test_kernel_cpu)

add_executable(test_kernel_shared_gpu test_kernel_shared_gpu.cpp)
target_link_libraries(test_kernel_shared_gpu PRIVATE
    test_helpers
    solvers
    doctest::doctest
)
target_include_directories(test_kernel_shared_gpu PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME test_kernel_shared_gpu COMMAND srun -n 1 ./test_kernel_shared_gpu)

add_executable(test_kernel_distributed_gpu test_kernel_distributed_gpu.cpp)
target_link_libraries(test_kernel_distributed_gpu PRIVATE
    test_helpers
    solvers
    doctest::doctest
)
target_include_directories(test_kernel_distributed_gpu PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    "/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include"
)
add_test(NAME test_kernel_distributed_gpu COMMAND prun -n 4 ./test_kernel_distributed_gpu)