# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.5)

project( example_trmm_mixed CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example example_trmm_mixed
add_executable( example_trmm_mixed example_trmm_mixed.cpp )
target_link_libraries( example_trmm_mixed PRIVATE tlapack Eigen3::Eigen )
