# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( performance_eigen CXX )

# Options
option( USE_MKL "Use MKL for Eigen" OFF )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# Load Eigen
find_package( Eigen3 REQUIRED )

# add the example performance_eigen
add_executable( performance_eigen performance_eigen.cpp )
target_link_libraries( performance_eigen PRIVATE Eigen3::Eigen )

# add the example performance_tlapack
add_executable( performance_tlapack performance_tlapack.cpp )
target_link_libraries( performance_tlapack PRIVATE tlapack Eigen3::Eigen )

# Load MKL
if( USE_MKL )
  set( BLA_VENDOR Intel10_64lp )
  find_package( BLAS REQUIRED )
  if( BLAS_FOUND )
    target_compile_definitions( performance_eigen PRIVATE "EIGEN_USE_MKL_ALL" )
    target_link_libraries( performance_eigen PRIVATE ${BLAS_LIBRARIES} )
  endif()
endif()