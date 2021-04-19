cmake_minimum_required(VERSION 3.1)

project(TestBLAS
    VERSION 0.1.2
    LANGUAGES "CXX")

# Version:
file(WRITE "version.txt" ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_VERSION})
message(STATUS "TestBLAS version " ${PROJECT_VERSION})

# Create cpp file for tests with corner cases
set(test_corner_cases_tables
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/blas_routines.csv"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/returnImmediately_corner_rules.csv"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/returnImmediately_corner_tests.csv"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/throwException_corner_rules.csv"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/throwException_corner_tests.csv"
    )
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp"
    COMMAND
        cd ${CMAKE_CURRENT_SOURCE_DIR}/scripts && ./wrapper_tests.py
        > "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp"
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/wrapper_tests.py"
    DEPENDS ${test_corner_cases_tables}
    VERBATIM)
add_custom_target(corner-cases
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp")

set(tester "tester")
file(GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp"
    )
add_executable(tester tests_main.cpp ${test_sources})
target_include_directories(tester PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Run tests
add_custom_target(run-tests
    COMMAND ./${tester}
    MAIN_DEPENDENCY ${tester}
    )