add_library(problem_spec INTERFACE)
target_include_directories(problem_spec INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(solver INTERFACE)
target_link_libraries(solver INTERFACE problem_spec)

# Profiling
add_library(kernel_slow_profile kernel_slow.cpp)
target_link_libraries(kernel_slow_profile PUBLIC problem_spec)
target_include_directories(kernel_slow_profile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(kernel_slow_profile PRIVATE PROFILE)

add_library(kernel_fast_profile kernel_fast.cpp)
target_link_libraries(kernel_fast_profile PUBLIC problem_spec)
target_include_directories(kernel_fast_profile PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/opt/rocm-7.1.0/include"
    "/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include"
)
target_compile_definitions(kernel_fast_profile PRIVATE PROFILE)

add_executable(heat_diffusion_profile main.cpp)
target_link_libraries(heat_diffusion_profile PRIVATE kernel_slow_profile kernel_fast_profile problem_spec solver)
target_include_directories(heat_diffusion_profile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/opt/rocm-7.1.0/include"
    "/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include"
)
target_compile_definitions(heat_diffusion_profile PRIVATE PROFILE)

# Solver
add_library(kernel_slow_solver kernel_slow.cpp)
target_link_libraries(kernel_slow_solver PUBLIC problem_spec)
target_include_directories(kernel_slow_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(kernel_fast_solver kernel_fast.cpp)
target_link_libraries(kernel_fast_solver PUBLIC problem_spec)
target_include_directories(kernel_fast_solver PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/opt/rocm-7.1.0/include"
    "/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include"
)

add_executable(heat_diffusion_solver main.cpp)
target_link_libraries(heat_diffusion_solver PRIVATE kernel_slow_solver kernel_fast_solver problem_spec solver)
target_include_directories(heat_diffusion_solver PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/opt/rocm-7.1.0/include"
    "/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include"
)