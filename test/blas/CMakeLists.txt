cmake_minimum_required(VERSION 3.1)

#-------------------------------------------------------------------------------
# Read project version
file(READ version.txt version_txt)

#-------------------------------------------------------------------------------
# Test BLAS project
project(TestBLAS
    VERSION ${version_txt}
    LANGUAGES CXX)
message(STATUS "TestBLAS version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# TODO: Define a `TBLASConfig.cmake` on the C++-based BLAS
set( tblas_include_dir "../../include" )

#-------------------------------------------------------------------------------
# Search for Catch2
find_package( Catch2 REQUIRED )
include(Catch)

#-------------------------------------------------------------------------------
# corner-cases: Generate C++ file for tests with corner cases
add_custom_target(corner-cases
    COMMAND
        cd "${CMAKE_CURRENT_SOURCE_DIR}/scripts" && ./wrapper_tests.py
        > "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp"
    )

#-------------------------------------------------------------------------------
# tester: Program for tests
file( GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp" )
add_executable( tester
    tests_main.cpp ${test_sources} )

target_compile_definitions( tester
    PRIVATE
        ${tlapack_defines} )

target_include_directories( tester
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${tblas_include_dir} )

target_link_libraries( tester
    PRIVATE
        Catch2::Catch2
        blaspp )

#-------------------------------------------------------------------------------
# Run tests
add_custom_target(run-tests
    COMMAND ./tester
    DEPENDS tester
    )

#-------------------------------------------------------------------------------
# Add tests to CTest
catch_discover_tests(tester)