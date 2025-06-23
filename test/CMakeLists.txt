# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

#-------------------------------------------------------------------------------
# Load Catch2
find_package( Catch2 3 QUIET )
if( NOT Catch2_FOUND )
  message( STATUS "Making Catch2 available..." )
  FetchContent_MakeAvailable( Catch2 )
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  message( STATUS "  Populated content: ${catch2_SOURCE_DIR}" )
else()
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
  message( STATUS "Found Catch2: ${Catch2_DIR}" )
endif()

#-------------------------------------------------------------------------------
# Load testsweeper if needed
if( BUILD_BLASPP_TESTS OR BUILD_LAPACKPP_TESTS )
  find_package( testsweeper 2021 QUIET )
  if( NOT testsweeper_FOUND )
    message( STATUS "Making testsweeper available..." )
    FetchContent_MakeAvailable( testsweeper )
    message( STATUS "  Populated content: ${testsweeper_SOURCE_DIR}" )
  else()
    message( STATUS "Found testsweeper: ${testsweeper_DIR}" )
  endif()
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
    message(FATAL_ERROR "The testBLAS submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules or use -D BUILD_testBLAS_TESTS=OFF and try again.")
  endif()

  set( TESTBLAS_BLAS tlapack CACHE STRING "C++ package used to access BLAS routines." )  
  add_subdirectory( testBLAS )

endif()
