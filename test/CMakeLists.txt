# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

include( "${TLAPACK_SOURCE_DIR}/cmake/FetchPackage.cmake" )

#-------------------------------------------------------------------------------
# Load Catch2
FetchPackage( "Catch2" "https://github.com/catchorg/Catch2.git" "v3.0.1" )
if( EXISTS "${Catch2_SOURCE_DIR}" )
  list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
else()
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()

#-------------------------------------------------------------------------------
# Build tests in the src directory
add_subdirectory(src)

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

#-------------------------------------------------------------------------------
# Build testBLAS tests
if( BUILD_testBLAS_TESTS )

  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/testBLAS/CMakeLists.txt")
    message(FATAL_ERROR "The testBLAS submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules or -Duse BUILD_testBLAS_TESTS=OFF and try again.")
  endif()

  add_subdirectory( testBLAS )

endif()

#-------------------------------------------------------------------------------
# Create a single-file tester with all tests

if( TLAPACK_BUILD_SINGLE_TESTER )

  file( GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp" )
  add_executable( tester ${test_sources} )

  if( BUILD_testBLAS_TESTS )
    file( GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/testBLAS/src/test_*.cpp" )
    target_sources( tester PRIVATE ${test_sources} )
  endif()

endif()
