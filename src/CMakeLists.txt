# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# # specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

#-------------------------------------------------------------------------------
# Library: libtblas_c
if( C_WRAPPERS OR Fortran_WRAPPERS )
  add_library( tblas_c tblas_cwrappers.cpp )
  target_link_libraries( tblas_c PUBLIC tblas )

  list( APPEND installable_libs tblas_c )
  if( C_WRAPPERS )
    install(
      FILES ${PROJECT_SOURCE_DIR}/include/tblas.h
      DESTINATION include )
  endif()
endif()

#-------------------------------------------------------------------------------
# Library: libtlapack_cblas
if( CBLAS_WRAPPERS )
  add_library( tlapack_cblas tblas_cwrappers.cpp )
  target_link_libraries( tlapack_cblas PUBLIC tblas )
  target_compile_definitions( tlapack_cblas PRIVATE BUILD_CBLAS )

  list( APPEND installable_libs tlapack_cblas )
  install(
    FILES ${PROJECT_SOURCE_DIR}/include/tlapack_cblas.h
    DESTINATION include )
endif()

#-------------------------------------------------------------------------------
# Library: libtblas_fortran
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

  # Fortran interface. Used to call C-based BLAS directly from a Fortran program.
  # The Fortran code needs to import this interface and link with the library blas
  configure_file(
    tblas.fi.in
    ${CMAKE_CURRENT_SOURCE_DIR}/tblas.fi @ONLY )

  # Constants module.
  configure_file( 
    blas/constants.f90.in 
    ${CMAKE_CURRENT_SOURCE_DIR}/blas/constants.f90 @ONLY )

  add_library( tblas_fortran
    tblas.f90
    blas/constants.f90
    blas/axpy.f90 )
  target_include_directories( tblas_fortran
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
  INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
    $<INSTALL_INTERFACE:lib> )
  target_link_libraries( tblas_fortran PUBLIC tblas_c )

  list( APPEND installable_libs tblas_fortran )
  list( APPEND installable_mods ${CMAKE_Fortran_MODULE_DIRECTORY}/tblas.mod )

  install(
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/tblas.fi
      ${CMAKE_CURRENT_SOURCE_DIR}/tblas.f90
    DESTINATION include )
  install(
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/blas/constants.f90
    DESTINATION include/blas )
endif()

set( installable_libs ${installable_libs} PARENT_SCOPE )
set( installable_mods ${installable_mods} PARENT_SCOPE )
