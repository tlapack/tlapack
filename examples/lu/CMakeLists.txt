# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.5)

project( lu CXX )
set(CMAKE_PREFIX_PATH "~/Documents/tlapack/")
set(CMAKE_CXX_STANDARD 20)
# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example lu
add_executable( example_lu example_lu.cpp )
target_link_libraries( example_lu PRIVATE tlapack )

#-------------------------------------------------------------------------------
# Search for MPFR library
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package( MPFR 2.3.1 QUIET )
find_package( GMP  4.2.1 QUIET )
if( MPFR_FOUND AND GMP_FOUND )

  target_compile_definitions( example_lu PRIVATE USE_MPFR )
  target_include_directories( example_lu PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES} "." )
  target_link_libraries( example_lu PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES} ) 
  
endif()