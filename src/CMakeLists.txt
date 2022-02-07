# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Options

set( blas_intt_list 
  int
  short
  long
  "long long"
  int8_t
  int16_t
  int32_t
  int64_t
  int_least8_t
  int_least16_t
  int_least32_t
  int_least64_t
  int_fast8_t
  int_fast16_t
  int_fast32_t
  int_fast64_t
  intmax_t
  intptr_t
  ptrdiff_t )

set( BLAS_INT_T int64_t CACHE STRING
  "Type of all non size-related integers in libtblas_c, libtlapack_cblas, and libtblas_fortran" )
set_property( CACHE BLAS_INT_T
  PROPERTY STRINGS ${blas_intt_list} )

set( BLAS_SIZE_T size_t CACHE STRING
  "Type of all size-relatedintegers in libtblas_c, libtlapack_cblas, and libtblas_fortran" )
set_property( CACHE BLAS_SIZE_T
  PROPERTY STRINGS ${blas_intt_list} size_t uint8_t uint16_t uint32_t uint64_t )

#-------------------------------------------------------------------------------
# Force options when using BLAS++ or LAPACK++ wrappers

if( USE_BLASPP_WRAPPERS OR USE_LAPACKPP_WRAPPERS )
  mark_as_advanced( FORCE BLAS_INT_T BLAS_SIZE_T )
  get_property( docString CACHE BLAS_INT_T PROPERTY HELPSTRING )
  set( BLAS_INT_T int64_t CACHE STRING "${docString}" FORCE )
  get_property( docString CACHE BLAS_SIZE_T PROPERTY HELPSTRING )
  set( BLAS_SIZE_T int64_t CACHE STRING "${docString}" FORCE )
else()
  mark_as_advanced( CLEAR BLAS_INT_T BLAS_SIZE_T )
endif()

#-------------------------------------------------------------------------------
# Library: libtblas_c
if( C_WRAPPERS OR Fortran_WRAPPERS )
  add_library( tblas_c tblas_cwrappers.cpp )
  target_link_libraries( tblas_c PUBLIC tblas )

  set( TBLAS_DEFINES "" )
  if( NOT BLAS_SIZE_T STREQUAL "" )
    string( APPEND TBLAS_DEFINES "#define BLAS_SIZE_T ${BLAS_SIZE_T}\n" )
  endif()
  if( NOT BLAS_INT_T STREQUAL "" )
    string( APPEND TBLAS_DEFINES "#define BLAS_INT_T ${BLAS_INT_T}\n" )
  endif()
  configure_file(
    ${PROJECT_SOURCE_DIR}/include/tblas.h.in
    ${PROJECT_BINARY_DIR}/include/tblas.h @ONLY )

  list( APPEND installable_libs tblas_c )
  if( C_WRAPPERS )
    install(
      FILES ${PROJECT_BINARY_DIR}/include/tblas.h
      DESTINATION include )
  endif()
endif()

#-------------------------------------------------------------------------------
# Library: libtlapack_cblas
if( CBLAS_WRAPPERS )
  add_library( tlapack_cblas tblas_cwrappers.cpp )
  target_link_libraries( tlapack_cblas PUBLIC tblas )
  target_compile_definitions( tlapack_cblas PRIVATE BUILD_CBLAS )
  
  set( TLAPACK_CBLAS_DEFINES "" )
  if( NOT BLAS_INT_T STREQUAL "" )
    string( APPEND TLAPACK_CBLAS_DEFINES "#define CBLAS_INT ${BLAS_INT_T}\n" )
  endif()
  configure_file(
    ${PROJECT_SOURCE_DIR}/include/tlapack_cblas.h.in
    ${PROJECT_BINARY_DIR}/include/tlapack_cblas.h @ONLY )

  list( APPEND installable_libs tlapack_cblas )
  install(
    FILES ${PROJECT_BINARY_DIR}/include/tlapack_cblas.h
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
    blas/axpy.f90
    blas/symm.f90
    blas/nrm2.f90 )
  target_include_directories( tblas_fortran
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
    $<INSTALL_INTERFACE:include> )
  target_link_libraries( tblas_fortran PUBLIC tblas_c )

  list( APPEND installable_libs tblas_fortran )
  list( APPEND installable_mods ${CMAKE_Fortran_MODULE_DIRECTORY}/tblas.mod )
  list( APPEND installable_mods ${CMAKE_Fortran_MODULE_DIRECTORY}/constants.mod )

  install(
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/tblas.fi
      ${CMAKE_CURRENT_SOURCE_DIR}/tblas.f90
    DESTINATION include )
  install(
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/blas/constants.f90
    DESTINATION include )
endif()

set( installable_libs ${installable_libs} PARENT_SCOPE )
set( installable_mods ${installable_mods} PARENT_SCOPE )
