# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( potrf CXX )

# Add the CMake directory for custom CMake modules
set(CMAKE_MODULE_PATH "cmake" ${CMAKE_MODULE_PATH})

if(LAPACK_DIR)
    find_package(LAPACK HINTS ${LAPACK_DIR})
else()
    find_package(LAPACK)
endif()

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example potrf
add_executable( example_potrf example_potrf.cpp )
target_link_libraries( example_potrf PRIVATE tlapack )

# Load LAPACKE if not using MKL
if( NOT USE_MKL )
  if(LAPACKE_DIR)
      find_package(LAPACKE HINTS ${LAPACKE_DIR})
  else()
      find_package(LAPACKE)
  endif()
  message( STATUS "LAPACKE: ${LAPACKE_LIBRARIES}" )
  target_link_libraries( example_potrf PRIVATE ${LAPACKE_LIBRARIES} )
else()
  target_compile_definitions( example_potrf PRIVATE "USE_MKL" )
  target_link_libraries( example_potrf PRIVATE ${LAPACK_LIBRARIES} )
endif()
