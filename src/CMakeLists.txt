add_library(problem_spec INTERFACE)
target_include_directories(problem_spec INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(solver INTERFACE)
target_link_libraries(solver INTERFACE problem_spec)

add_library(kernel_slow kernel_slow.cpp)
target_link_libraries(kernel_slow PUBLIC problem_spec)
target_include_directories(kernel_slow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main PRIVATE kernel_slow problem_spec solver)