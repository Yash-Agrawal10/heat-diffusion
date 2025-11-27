add_library(problem_spec INTERFACE)
target_include_directories(problem_spec INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(solver INTERFACE)
target_link_libraries(solver INTERFACE problem_spec)

# Profiling
add_library(kernels_profile kernel_slow.cpp kernel_fast.cpp)
target_link_libraries(kernels_profile PUBLIC problem_spec)
target_include_directories(kernels_profile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(kernels_profile PRIVATE PROFILE)

add_executable(heat_diffusion_profile main.cpp)
target_link_libraries(heat_diffusion_profile PRIVATE kernels_profile problem_spec solver)
target_compile_definitions(heat_diffusion_profile PRIVATE PROFILE)

# Non-profiling
add_library(kernels kernel_slow.cpp kernel_fast.cpp)
target_link_libraries(kernels PUBLIC problem_spec)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(heat_diffusion main.cpp)
target_link_libraries(heat_diffusion PRIVATE kernels problem_spec solver)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})