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
target_include_directories(kernel_fast_profile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(kernel_fast_profile PRIVATE PROFILE)

add_executable(heat_diffusion_profile main.cpp)
target_link_libraries(heat_diffusion_profile PRIVATE kernel_slow_profile kernel_fast_profile problem_spec solver)
target_compile_definitions(heat_diffusion_profile PRIVATE PROFILE)

# Non-profiling
add_library(kernel_slow kernel_slow.cpp)
target_link_libraries(kernel_slow PUBLIC problem_spec)
target_include_directories(kernel_slow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(kernel_fast kernel_fast.cpp)
target_link_libraries(kernel_fast PUBLIC problem_spec)
target_include_directories(kernel_fast PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(heat_diffusion main.cpp)
target_link_libraries(heat_diffusion PRIVATE kernel_slow kernel_fast problem_spec solver)