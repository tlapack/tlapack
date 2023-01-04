# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( potrf CXX )

# Add the CMake directory for custom CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Load LAPACK
find_package( LAPACK REQUIRED )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example potrf
add_executable( example_potrf example_potrf.cpp )
target_link_libraries( example_potrf PRIVATE tlapack )

# Load LAPACKE if not using MKL
if( NOT USE_MKL )
  find_package( LAPACKE REQUIRED )
  target_include_directories( example_potrf PRIVATE ${LAPACKE_INCLUDE_DIRS} )
  target_link_libraries( example_potrf PRIVATE ${LAPACKE_LIBRARIES} )
else()
  target_compile_definitions( example_potrf PRIVATE "USE_MKL" )
  target_link_libraries( example_potrf PRIVATE ${LAPACK_LIBRARIES} )
endif()
