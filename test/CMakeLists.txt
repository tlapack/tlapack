# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

include( "${TLAPACK_SOURCE_DIR}/cmake/FetchPackage.cmake" )

#-------------------------------------------------------------------------------
# Build BLAS++ tests
if( BUILD_BLASPP_TESTS )
  add_subdirectory( blaspp )
endif()

#-------------------------------------------------------------------------------
# Build LAPACK++ tests
if( BUILD_LAPACKPP_TESTS )
  add_subdirectory( lapackpp )
endif()

#########################
## Tests from <T>LAPACK :

#-------------------------------------------------------------------------------
# Load Catch2
FetchPackage( "Catch2" "https://github.com/catchorg/Catch2.git" "v2.13.1" )

#-------------------------------------------------------------------------------
# Add folder with Catch.cmake to the CMAKE_MODULE_PATH
set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_SOURCE_DIR}/extras;${Catch2_SOURCE_DIR}/contrib" )

add_subdirectory(src)

#-------------------------------------------------------------------------------
# tester: Program for tests

if( TLAPACK_BUILD_SINGLE_TESTER )# Test sources
  file( GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp" )
  add_executable( tester tests_main.cpp ${test_sources} )
  set_target_properties( tester
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
else()
  add_executable( tester tests_main.cpp )
  set_target_properties( tester
    PROPERTIES
      OUTPUT_NAME "tester_testBLAS"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test" )
endif()

# Load testBLAS tests
FetchPackage( "testBLAS" "https://github.com/tlapack/testBLAS.git" "master" )

target_include_directories( tester PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" )
target_link_libraries( tester PRIVATE Catch2::Catch2 tlapack )

if( TEST_MPFR )
  target_include_directories( tester PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES} )
  target_link_libraries( tester PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
endif()

# Add tests to CTest
include(Catch)
catch_discover_tests(tester)
