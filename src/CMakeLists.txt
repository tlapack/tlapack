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

set( TLAPACK_INT_T int64_t CACHE STRING
  "Type of all non size-related integers in libtlapack_c, libtlapack_cblas, and libtlapack_fortran" )
set_property( CACHE TLAPACK_INT_T
  PROPERTY STRINGS ${blas_intt_list} )

set( TLAPACK_SIZE_T size_t CACHE STRING
  "Type of all size-relatedintegers in libtlapack_c, libtlapack_cblas, and libtlapack_fortran" )
set_property( CACHE TLAPACK_SIZE_T
  PROPERTY STRINGS ${blas_intt_list} size_t uint8_t uint16_t uint32_t uint64_t )

#-------------------------------------------------------------------------------
# Force options when using BLAS++ or LAPACK++ wrappers

if( USE_LAPACKPP_WRAPPERS )
  mark_as_advanced( FORCE TLAPACK_INT_T TLAPACK_SIZE_T )
  get_property( docString CACHE TLAPACK_INT_T PROPERTY HELPSTRING )
  set( TLAPACK_INT_T int64_t CACHE STRING "${docString}" FORCE )
  get_property( docString CACHE TLAPACK_SIZE_T PROPERTY HELPSTRING )
  set( TLAPACK_SIZE_T int64_t CACHE STRING "${docString}" FORCE )
else()
  mark_as_advanced( CLEAR TLAPACK_INT_T TLAPACK_SIZE_T )
endif()

#-------------------------------------------------------------------------------
# Library: libtlapack_c
if( C_WRAPPERS OR Fortran_WRAPPERS )
  add_library( tlapack_c tlapack_cwrappers.cpp )
  target_link_libraries( tlapack_c PUBLIC tlapack )

  set( TLAPACK_DEFINES "" )
  if( NOT TLAPACK_SIZE_T STREQUAL "" )
    string( APPEND TLAPACK_DEFINES "#define TLAPACK_SIZE_T ${TLAPACK_SIZE_T}\n" )
  endif()
  if( NOT TLAPACK_INT_T STREQUAL "" )
    string( APPEND TLAPACK_DEFINES "#define TLAPACK_INT_T ${TLAPACK_INT_T}\n" )
  endif()
  configure_file(
    ${PROJECT_SOURCE_DIR}/include/tlapack.h.in
    ${PROJECT_BINARY_DIR}/include/tlapack.h @ONLY )

  list( APPEND installable_libs tlapack_c )
  if( C_WRAPPERS )
    install(
      FILES ${PROJECT_BINARY_DIR}/include/tlapack.h
      DESTINATION include )
  endif()
endif()

#-------------------------------------------------------------------------------
# Library: libtlapack_cblas
if( CBLAS_WRAPPERS )
  add_library( tlapack_cblas tlapack_cwrappers.cpp )
  target_link_libraries( tlapack_cblas PUBLIC tlapack )
  target_compile_definitions( tlapack_cblas PRIVATE BUILD_CBLAS )
  
  set( TLAPACK_CBLAS_DEFINES "" )
  if( NOT TLAPACK_INT_T STREQUAL "" )
    string( APPEND TLAPACK_CBLAS_DEFINES "#define CBLAS_INT ${TLAPACK_INT_T}\n" )
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
# Library: libtlapack_fortran
if( Fortran_WRAPPERS )

  # Integer types mapped to the ISO_FORTRAN_ENV module
  if( (${TLAPACK_SIZE_T} IN_LIST blas_intt_list) OR
      (${TLAPACK_SIZE_T} STREQUAL "size_t") )
    string(REPLACE " " "_" FTLAPACK_SIZE_T_0 ${TLAPACK_SIZE_T})
    set( FTLAPACK_SIZE_T "c_${FTLAPACK_SIZE_T_0}" CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension"
    FORCE )
  else()
    string(REGEX REPLACE "^u" "" FTLAPACK_SIZE_T_0 ${TLAPACK_SIZE_T})
    if( ${FTLAPACK_SIZE_T_0} IN_LIST blas_intt_list )
      string(REPLACE " " "_" FTLAPACK_SIZE_T_0 ${FTLAPACK_SIZE_T_0})
      set( FTLAPACK_SIZE_T "c_${FTLAPACK_SIZE_T_0}" CACHE STRING
      "Integer type for array sizes, array indexes, and matrix leading dimension"
      FORCE )
    else()
      set( FTLAPACK_SIZE_T "" CACHE STRING
      "Integer type for array sizes, array indexes, and matrix leading dimension" )
    endif()
  endif()
  mark_as_advanced( FTLAPACK_SIZE_T )

  # Integer types mapped to the ISO_FORTRAN_ENV module
  if( ${TLAPACK_INT_T} IN_LIST blas_intt_list )
    string(REPLACE " " "_" FTLAPACK_INT_T_0 ${TLAPACK_INT_T})
    set( FTLAPACK_INT_T "c_${FTLAPACK_INT_T_0}" CACHE STRING
    "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension"
    FORCE )
  else()
    set( FTLAPACK_INT_T "" CACHE STRING
    "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension" )
  endif()
  mark_as_advanced( FTLAPACK_INT_T )

  # Fortran interface. Used to call C-based BLAS directly from a Fortran program.
  # The Fortran code needs to import this interface and link with the library blas
  configure_file(
    tlapack.fi.in
    ${CMAKE_CURRENT_SOURCE_DIR}/tlapack.fi @ONLY )

  # Constants module.
  configure_file( 
    base/constants.f90.in 
    ${CMAKE_CURRENT_SOURCE_DIR}/base/constants.f90 @ONLY )

  add_library( tlapack_fortran
    tlapack.f90
    base/constants.f90
    blas/axpy.f90
    blas/symm.f90
    blas/nrm2.f90 )
  target_include_directories( tlapack_fortran
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
    $<INSTALL_INTERFACE:include/tlapack/fortran> )
  target_link_libraries( tlapack_fortran PUBLIC tlapack_c )

  list( APPEND installable_libs tlapack_fortran )
  list( APPEND installable_mods ${CMAKE_Fortran_MODULE_DIRECTORY}/tlapack.mod )
  list( APPEND installable_mods ${CMAKE_Fortran_MODULE_DIRECTORY}/constants.mod )

  install(
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/tlapack.fi
      ${CMAKE_CURRENT_SOURCE_DIR}/tlapack.f90
    DESTINATION include )
  install(
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/base/constants.f90
    DESTINATION include/tlapack/base )
endif()

set( installable_libs ${installable_libs} PARENT_SCOPE )
set( installable_mods ${installable_mods} PARENT_SCOPE )
