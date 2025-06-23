# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.5)

project( gemm CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example gemm
add_executable( example_gemm example_gemm.cpp )
target_link_libraries( example_gemm PRIVATE tlapack )

#-------------------------------------------------------------------------------
# Search for MPFR library
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package( MPFR 2.3.1 QUIET )
find_package( GMP  4.2.1 QUIET )
if( MPFR_FOUND AND GMP_FOUND )

  target_compile_definitions( example_gemm PRIVATE USE_MPFR )
  target_include_directories( example_gemm PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES} "." )
  target_link_libraries( example_gemm PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES} ) 
  
endif()