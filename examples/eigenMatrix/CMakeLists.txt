# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( eigenMatrix CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

find_package( Eigen3 REQUIRED )

# add the example eigenMatrix
add_executable( example_eigenMatrix example_eigenMatrix.cpp )
target_link_libraries( example_eigenMatrix PRIVATE tlapack Eigen3::Eigen )
