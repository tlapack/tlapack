cmake_minimum_required(VERSION 3.1)

# Read project version:
file(READ version.txt version_txt)

#
# Test BLAS project
#
project(TestBLAS
    VERSION ${version_txt}
    LANGUAGES CXX)

message(STATUS "TestBLAS version " ${PROJECT_VERSION})

#
# Include
#

# BLAS 'blas.hpp' header file in C++
find_path(blas_inc NAMES "blas.hpp"
    HINTS ${CMAKE_SOURCE_DIR}/include)
if(blas_inc)
    message(STATUS "blas.hpp => " ${blas_inc})
    include_directories( ${blas_inc} )
else()
    message(STATUS "blas.hpp was NOT FOUND")
endif(blas_inc)

# CATCH2 'catch.hpp' header file
find_path(catch_inc NAMES "catch.hpp"
    HINTS ${CMAKE_SOURCE_DIR}/thirdparty/include/catch2)
if(catch_inc)
    message(STATUS "catch.hpp => " ${catch_inc})
    include_directories( ${catch_inc} )
else()
    message(STATUS "catch.hpp was NOT FOUND")
endif(catch_inc)

#
# Targets
#

# C++ file for tests with corner cases
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
add_custom_target(corner-cases ALL
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp")

# Create tester 
set(tester "tester")
file(GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp"
    )
add_executable(tester tests_main.cpp ${test_sources})
target_include_directories(tester PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Run tests
add_custom_target(run-tests
    COMMAND ./${tester}
    DEPENDS ${tester}
    )

#
# CTest
#

find_package(Catch2 REQUIRED
    HINTS ${CMAKE_SOURCE_DIR}/thirdparty)
target_link_libraries(tester PRIVATE Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(tester)