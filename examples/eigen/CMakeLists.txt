# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.5)

project( eigen CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

find_package( Eigen3 REQUIRED )

# add the example eigen
add_executable( example_eigen example_eigen.cpp )
target_link_libraries( example_eigen PRIVATE tlapack Eigen3::Eigen )
