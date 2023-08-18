# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( create_float_library CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

find_package( Eigen3 REQUIRED )

# Create the library
add_library( tlapack_EigenMatrixXf STATIC tlapack_EigenMatrixXf.cpp )
target_link_libraries( tlapack_EigenMatrixXf PUBLIC Eigen3::Eigen tlapack )

# add the example eigen
add_executable( example_create_float_library main.cpp )
target_link_libraries( example_create_float_library PRIVATE tlapack_EigenMatrixXf )
