# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( starpu CXX )
set(CMAKE_CXX_STANDARD 20)

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# Load StarPU
find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.4)

# add the example LU
add_executable( example_starpu_lu example_lu.cpp )
target_include_directories( example_starpu_lu PRIVATE ${STARPU_INCLUDE_DIRS} )
target_link_directories( example_starpu_lu PRIVATE ${STARPU_STATIC_LIBRARY_DIRS} )
target_link_libraries( example_starpu_lu PRIVATE tlapack ${STARPU_STATIC_LIBRARIES} )

# add the example GEMM
add_executable( example_starpu_gemm example_gemm.cpp )
target_include_directories( example_starpu_gemm PRIVATE ${STARPU_INCLUDE_DIRS} )
target_link_directories( example_starpu_gemm PRIVATE ${STARPU_STATIC_LIBRARY_DIRS} )
target_link_libraries( example_starpu_gemm PRIVATE tlapack ${STARPU_STATIC_LIBRARIES} )

# add the example POTRF
add_executable( example_starpu_potrf example_potrf.cpp )
target_include_directories( example_starpu_potrf PRIVATE ${STARPU_INCLUDE_DIRS} )
target_link_directories( example_starpu_potrf PRIVATE ${STARPU_STATIC_LIBRARY_DIRS} )
target_link_libraries( example_starpu_potrf PRIVATE tlapack ${STARPU_STATIC_LIBRARIES} )

# Find LAPACK and test if LAPACK_LIBRARIES contains MKL
find_package( LAPACK REQUIRED )
list( FIND LAPACK_LIBRARIES "mkl" MKL_FOUND ) 

# add the example using an optimized LAPACK library for comparison
add_executable( example_starpu_lapack example_lapack.cpp )
if( MKL_FOUND GREATER -1 )
  target_compile_definitions( example_starpu_lapack PRIVATE "USE_MKL" )
  target_link_libraries( example_starpu_lapack PRIVATE ${LAPACK_LIBRARIES} )
else()
  find_package( LAPACKE REQUIRED )
  target_include_directories( example_starpu_lapack PRIVATE ${LAPACKE_INCLUDE_DIRS} )
  target_link_libraries( example_starpu_lapack PRIVATE ${LAPACKE_LIBRARIES} )
endif()
