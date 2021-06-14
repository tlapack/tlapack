# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# # specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

#-------------------------------------------------------------------------------
# Library: libblas
if( C_WRAPPERS OR Fortran_WRAPPERS )
  add_library(blas cwrappers_tblas.cpp)
  target_link_libraries( blas PUBLIC tblas )

  list( APPEND installable_libs blas )
  if( C_WRAPPERS )
    list( APPEND installable_files
      ${PROJECT_SOURCE_DIR}/include/blas.h )
  endif()
endif()

#-------------------------------------------------------------------------------
# Library: libcblas
if( CBLAS_WRAPPERS )
  add_library(cblas cwrappers_tblas.cpp)
  target_link_libraries( cblas PUBLIC tblas )
  target_compile_definitions( cblas PRIVATE BUILD_CBLAS )

  list( APPEND installable_libs cblas )
  list( APPEND installable_files
    ${PROJECT_SOURCE_DIR}/include/cblas.h )
endif()

#-------------------------------------------------------------------------------
# Library: Fortran wrappers for TBLAS
if( Fortran_WRAPPERS )

  # Integer types mapped to the ISO_FORTRAN_ENV module
  if( (${BLAS_SIZE_T} IN_LIST blas_intt_list) OR
      (${BLAS_SIZE_T} STREQUAL "size_t") )
    string(REPLACE " " "_" FBLAS_SIZE_T_0 ${BLAS_SIZE_T})
    set( FBLAS_SIZE_T "c_${FBLAS_SIZE_T_0}" CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension"
    FORCE )
  else()
    string(REGEX REPLACE "^u" "" FBLAS_SIZE_T_0 ${BLAS_SIZE_T})
    if( ${FBLAS_SIZE_T_0} IN_LIST blas_intt_list )
      string(REPLACE " " "_" FBLAS_SIZE_T_0 ${FBLAS_SIZE_T_0})
      set( FBLAS_SIZE_T "c_${FBLAS_SIZE_T_0}" CACHE STRING
      "Integer type for array sizes, array indexes, and matrix leading dimension"
      FORCE )
    else()
      set( FBLAS_SIZE_T "" CACHE STRING
      "Integer type for array sizes, array indexes, and matrix leading dimension" )
    endif()
  endif()
  mark_as_advanced( FBLAS_SIZE_T )

  # Integer types mapped to the ISO_FORTRAN_ENV module
  if( ${BLAS_INT_T} IN_LIST blas_intt_list )
    string(REPLACE " " "_" FBLAS_INT_T_0 ${BLAS_INT_T})
    set( FBLAS_INT_T "c_${FBLAS_INT_T_0}" CACHE STRING
    "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension"
    FORCE )
  else()
    set( FBLAS_INT_T "" CACHE STRING
    "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension" )
  endif()
  mark_as_advanced( FBLAS_INT_T )

  # Choose the number of bytes for the integer type
  # TODO: Improve this part
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/getNumBytes.c.in
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/getNumBytes.c @ONLY )
  execute_process (
    COMMAND bash -c "${CMAKE_CXX_COMPILER} getNumBytes.c -o getNumBytes${CMAKE_EXECUTABLE_SUFFIX} && ./getNumBytes${CMAKE_EXECUTABLE_SUFFIX}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils"
    RESULT_VARIABLE FBLAS_INT_BYTES
  )

  # Truncate FBLAS_INT_BYTES to a value between 1 and 8
  # TODO: Improve this part
  if( (${FBLAS_INT_BYTES} GREATER_EQUAL 1) AND (${FBLAS_INT_BYTES} LESS_EQUAL 8) )
    message( STATUS "Fortran wrappers use integers of ${FBLAS_INT_BYTES} bytes" )
  else()
    set( FBLAS_INT_BYTES "8" )
    message( STATUS "Fortran wrappers use integers of ${FBLAS_INT_BYTES} bytes" )
  endif()

  # Fortran interface. Used to call C-based BLAS directly from a Fortran program.
  # The Fortran code needs to import this interface and link with the library blas
  configure_file( blas.fi.in ${PROJECT_SOURCE_DIR}/include/blas.fi @ONLY )

  # Fortran library. This library can be linked directly to a Fortran code.
  configure_file( fortranwrappers_tblas.f90.in ${CMAKE_CURRENT_SOURCE_DIR}/fortranwrappers_tblas.f90 @ONLY )

  # Fortran module. Translates the C-based BLAS interface into Fortran 90.
  # The Fortran code needs to load this interface and link with the library blas
  configure_file( blas.f90.in ${PROJECT_SOURCE_DIR}/include/blas.f90 @ONLY )

  add_library(fortranwrappers_tblas fortranwrappers_tblas.f90)
  target_link_libraries( fortranwrappers_tblas PUBLIC blas )

  list( APPEND installable_libs fortranwrappers_tblas )
  list( APPEND installable_files
    ${PROJECT_SOURCE_DIR}/include/blas.fi
    ${PROJECT_SOURCE_DIR}/include/blas.f90 )
endif()
