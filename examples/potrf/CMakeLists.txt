# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( potrf CXX )

find_package( LAPACK REQUIRED )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example potrf
add_executable( example_potrf example_potrf.cpp )
target_link_libraries( example_potrf PRIVATE tlapack )
target_link_libraries( example_potrf PRIVATE "-llapacke" ${LAPACK_LIBRARIES} )
