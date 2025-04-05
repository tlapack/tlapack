# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( extreme_interoperability CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# Load Eigen and mdspan
find_package( Eigen3 REQUIRED )
find_package( mdspan REQUIRED )

# add the example example_interoperability
add_executable( example_interoperability example_interoperability.cpp )
target_link_libraries( example_interoperability PRIVATE tlapack std::mdspan Eigen3::Eigen )
