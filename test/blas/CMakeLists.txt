# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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
# T-BLAS
if( NOT TARGET tblas )
  # TODO: Do something when target is not found
endif()

#-------------------------------------------------------------------------------
# Search for Catch2
find_package( Catch2 REQUIRED )
include(Catch)

#-------------------------------------------------------------------------------
# corner-cases: Generate C++ file for tests with corner cases
add_custom_target( corner-cases
COMMAND
    cd "${CMAKE_CURRENT_SOURCE_DIR}/scripts" && ./wrapper_tests.py
    > "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp" )

#-------------------------------------------------------------------------------
# Test sources
file( GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp" )

# #-------------------------------------------------------------------------------
# # Test library
# add_library( testBLAS ${test_sources} )

# target_compile_definitions( testBLAS PUBLIC ${tlapack_defs} )
# target_include_directories( testBLAS
#     PUBLIC "${PROJECT_SOURCE_DIR}/include" ${tblas_include_dir} )

# if( USE_BLASPP )
#     target_link_libraries( testBLAS PUBLIC blaspp )
# endif()
# target_link_libraries( testBLAS PUBLIC Catch2::Catch2 )

# set_target_properties( testBLAS
#     PROPERTIES
#     ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )

#-------------------------------------------------------------------------------
# tester: Program for tests

# # add_subdirectory( src )
# add_executable( tester tests_main.cpp )
# target_link_libraries( tester PRIVATE testBLAS )

add_executable( tester tests_main.cpp ${test_sources} )
target_include_directories( tester PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" )
target_link_libraries( tester PRIVATE Catch2::Catch2 tblas )

set_target_properties( tester
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

#-------------------------------------------------------------------------------
# Add tests to CTest
catch_discover_tests(tester)