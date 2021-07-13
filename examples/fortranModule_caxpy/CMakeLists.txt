# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( fortranModule_caxpy C CXX Fortran )

# Load <T>LAPACK
if( NOT TARGET tblas_fortran )
  find_package( tlapack REQUIRED )
elseif( NOT EXISTS tlapack_DIR )
  set( tlapack_DIR ${CMAKE_SOURCE_DIR} )
endif()

# add the example fortranModule_saxpy
add_executable( example_fortranModule_caxpy example_fortranModule_caxpy.f90 )
target_link_libraries( example_fortranModule_caxpy PRIVATE tblas_fortran )
