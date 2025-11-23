add_library(solver solver.cpp)
target_include_directories(solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(heat_diffusion main.cpp)
target_link_libraries(heat_diffusion PRIVATE solver)