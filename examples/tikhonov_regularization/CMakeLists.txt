# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.5)

project( tikhonov_regularization CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example tikhonov_regularization
add_executable( example_tikhonov_regularization example_tikhonov_regularization.cpp )
target_link_libraries( example_tikhonov_regularization PRIVATE tlapack )
