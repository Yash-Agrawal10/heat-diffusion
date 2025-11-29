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
)

add_executable(test_analytic test_analytic.cpp)
target_link_libraries(test_analytic PRIVATE
    test_helpers
    doctest::doctest
)
add_test(NAME test_analytic COMMAND srun ./test_analytic)

add_executable(test_kernel_slow test_kernel_slow.cpp)
target_link_libraries(test_kernel_slow PRIVATE
    kernel_slow_solver
    problem_spec
    solver
    test_helpers
    doctest::doctest
)
add_test(NAME test_kernel_slow COMMAND srun ./test_kernel_slow)

add_executable(test_kernel_fast test_kernel_fast.cpp)
target_link_libraries(test_kernel_fast PRIVATE
    kernel_fast_solver
    problem_spec
    solver
    test_helpers
    doctest::doctest
)
add_test(NAME test_kernel_fast COMMAND srun ./test_kernel_fast)