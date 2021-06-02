# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of T-LAPACK.
# T-LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# # specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

#-------------------------------------------------------------------------------
# Library: libblas
add_library(blas cwrappers_tblas.cpp)
target_link_libraries( blas PUBLIC tblas )

#-------------------------------------------------------------------------------
# Library: libcblas
add_library(cblas cwrappers_tblas.cpp)
target_link_libraries( blas PUBLIC tblas )
target_compile_definitions( cblas PRIVATE BUILD_CBLAS )

#-------------------------------------------------------------------------------
# Library: Fortran wrappers for TBLAS

# Integer types mapped to the ISO_FORTRAN_ENV module
if( (${BLAS_SIZE_T} IN_LIST blas_intt_list) OR
    (${BLAS_SIZE_T} STREQUAL "size_t") )
  string(REPLACE " " "_" CBLAS_SIZE_T_0 ${BLAS_SIZE_T})
  set( CBLAS_SIZE_T "c_${CBLAS_SIZE_T_0}" CACHE STRING
  "Integer type for array sizes, array indexes, and matrix leading dimension"
  FORCE )
else()
  string(REGEX REPLACE "^u" "" CBLAS_SIZE_T_0 ${BLAS_SIZE_T})
  if( ${CBLAS_SIZE_T_0} IN_LIST blas_intt_list )
    string(REPLACE " " "_" CBLAS_SIZE_T_0 ${CBLAS_SIZE_T_0})
    set( CBLAS_SIZE_T "c_${CBLAS_SIZE_T_0}" CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension"
    FORCE )
  else()
    set( CBLAS_SIZE_T "" CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension" )
  endif()
endif()

# Integer types mapped to the ISO_FORTRAN_ENV module
if( ${BLAS_INT_T} IN_LIST blas_intt_list )
  string(REPLACE " " "_" CBLAS_INT_T_0 ${BLAS_INT_T})
  set( CBLAS_INT_T "c_${CBLAS_INT_T_0}" CACHE STRING
  "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension"
  FORCE )
else()
  set( CBLAS_INT_T "" CACHE STRING
  "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension" )
endif()

# Fortran interface. Used to call C-based BLAS directly from a Fortran program.
# The Fortran code needs to import this interface and link with the library blas
configure_file( cblas.fi.in ${PROJECT_SOURCE_DIR}/include/cblas.fi @ONLY )

# Fortran library. This library can be linked directly to a Fortran code.
configure_file( fortranwrappers_tblas.f90.in ${CMAKE_CURRENT_SOURCE_DIR}/fortranwrappers_tblas.f90 @ONLY )

# Fortran module. Translates the C-based BLAS interface into Fortran 90.
# The Fortran code needs to load this interface and link with the library blas
configure_file( blas.f90.in ${PROJECT_SOURCE_DIR}/include/blas.f90 @ONLY )

add_library(fortranwrappers_tblas fortranwrappers_tblas.f90)
target_link_libraries( fortranwrappers_tblas PUBLIC blas )

#-------------------------------------------------------------------------------
# Common configurations

set_target_properties( blas cblas fortranwrappers_tblas
PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )