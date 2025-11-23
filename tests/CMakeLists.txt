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
target_link_libraries(test_analytic
    test_helpers
    doctest::doctest
)

add_test(NAME test_analytic COMMAND test_analytic)