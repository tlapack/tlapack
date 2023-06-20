# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( eigenvalues CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# Check if Fortran is available
include(CheckLanguage)
check_language(Fortran)
if( CMAKE_Fortran_COMPILER )
  enable_language(Fortran)
endif()

# Find LAPACK
find_package( LAPACK QUIET )
file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f90")

# add the example example_eigenvalues
add_executable( example_eigenvalues example_eigenvalues.cpp )
target_link_libraries( example_eigenvalues PRIVATE tlapack )
if( LAPACK_FOUND AND CMAKE_Fortran_COMPILER )
  target_compile_definitions( example_eigenvalues PRIVATE USE_LAPACK )
  target_sources(example_eigenvalues PRIVATE ${SRC_FILES})
  target_link_libraries( example_eigenvalues PRIVATE ${LAPACK_LIBRARIES} )
endif()

# add the example profile_aed
add_executable( profile_aed profile_aed.cpp )
target_link_libraries( profile_aed PRIVATE tlapack )
if( LAPACK_FOUND AND CMAKE_Fortran_COMPILER )
  target_compile_definitions( profile_aed PRIVATE USE_LAPACK )
  target_sources(profile_aed PRIVATE ${SRC_FILES})
  target_link_libraries( profile_aed PRIVATE ${LAPACK_LIBRARIES} )
endif()
