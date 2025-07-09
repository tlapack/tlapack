# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.5)

project( laed4 CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example example_laed4
add_executable( example_laed4 example_laed4.cpp )
target_link_libraries( example_laed4 PRIVATE tlapack )

# add the example example_laed4_brian
add_executable( example_laed4_brian example_laed4_brian.cpp )
target_link_libraries( example_laed4_brian PRIVATE tlapack )

# add the example example_laed4_sorensen_tang
add_executable( example_laed4_sorensen_tang example_laed4_sorensen_tang.cpp )
target_link_libraries( example_laed4_sorensen_tang PRIVATE tlapack )

# add the example example_laed4_sorensen_tang
add_executable( example_laed4_gragg example_laed4_gragg.cpp )
target_link_libraries( example_laed4_gragg PRIVATE tlapack )
