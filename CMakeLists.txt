# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of T-LAPACK.
# T-LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.18)
# VERSION 3.18: set_source_files_properties( PROPERTIES Fortran_PREPROCESS ON )

#-------------------------------------------------------------------------------
# Read project version
file(READ version.txt version_txt)

#-------------------------------------------------------------------------------
# T-LAPACK project
project(TLAPACK
  VERSION ${version_txt}
  LANGUAGES CXX C Fortran)
message(STATUS "T-LAPACK version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# T-LAPACK and T-BLAS template library

add_library( tblas src/blas/exception.cpp )

add_library( tlapack INTERFACE )
target_link_libraries( tlapack INTERFACE tblas )

set_target_properties( tlapack tblas
PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )

#-------------------------------------------------------------------------------
# Options
option( USE_BLASPP_WRAPPERS   "Use BLAS++ wrappers"   OFF )
option( USE_BLASPP_TEMPLATES  "Use BLAS++ templates"  OFF )
option( USE_LAPACKPP_WRAPPERS "Use LAPACK++ wrappers" OFF )
option( BUILD_EXAMPLES        "Build examples"        ON  )
option( USE_MPFR              "Use MPFR C++ library (http://www.holoborodko.com/pavel/mpfr/)"  OFF )

#-------------------------------------------------------------------------------
# Modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#-------------------------------------------------------------------------------
# Search for BLAS++ library if it is needed

if( USE_BLASPP_WRAPPERS OR USE_BLASPP_TEMPLATES )
  set(USE_BLASPP ON)
  find_package( blaspp )
  if(blaspp_FOUND)
    message( STATUS "  BLAS++ found in ${blaspp_DIR}" )
  else()
    message( FATAL_ERROR "  BLAS++ not found." )
  endif()
  target_link_libraries( tblas PUBLIC blaspp )
endif()

if(USE_BLASPP_WRAPPERS)
  target_compile_definitions( tblas PUBLIC USE_BLASPP_WRAPPERS )
endif()

if(USE_BLASPP_TEMPLATES)
  target_compile_definitions( tblas PUBLIC USE_BLASPP_TEMPLATES )
endif()

#-------------------------------------------------------------------------------
# Search for LAPACK++ library if it is needed
if( USE_LAPACKPP_WRAPPERS )
  find_package( lapackpp )
  if(lapackpp_FOUND)
    message( STATUS "  LAPACK++ found in ${lapackpp_DIR}" )
  else()
    message( FATAL_ERROR "  LAPACK++ not found." )
  endif()
  target_compile_definitions( tlapack INTERFACE USE_LAPACKPP_WRAPPERS )
  target_link_libraries( tlapack INTERFACE lapackpp )
endif()

#-------------------------------------------------------------------------------
# Search for MPFR library if it is needed
if( USE_MPFR )
  find_package( GMP REQUIRED )
  find_package( MPFR REQUIRED )
  target_compile_definitions( tblas PUBLIC USE_MPFR )
  target_include_directories( tblas PUBLIC ${MPFR_INCLUDES} ${GMP_INCLUDES} )
  target_link_libraries( tblas PUBLIC ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
endif()

#-------------------------------------------------------------------------------
# Options for BLAS_SIZE_T

if( USE_BLASPP OR USE_LAPACKPP_WRAPPERS )
  set( BLAS_SIZE_T int64_t )
else()
  set( BLAS_SIZE_T size_t CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension" )
  set_property( CACHE BLAS_SIZE_T
    PROPERTY STRINGS
    intmax_t size_t 
    int8_t  uint8_t
    int16_t uint16_t
    int32_t uint32_t
    int64_t uint64_t )
endif()

if( NOT BLAS_SIZE_T STREQUAL "" )
  target_compile_definitions( tblas PUBLIC "BLAS_SIZE_T=${BLAS_SIZE_T}" )
endif()

#-------------------------------------------------------------------------------
# Options for BLAS_INT_T

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
  
if( USE_BLASPP OR USE_LAPACKPP_WRAPPERS )
  set( BLAS_INT_T int64_t )
else()
  set( BLAS_INT_T int64_t CACHE STRING
    "Integer type for all variables except array sizes and matrix leading dimension" )
  set_property( CACHE BLAS_INT_T PROPERTY STRINGS ${blas_intt_list} )
endif()

if( NOT BLAS_INT_T STREQUAL "" )
  target_compile_definitions( tblas PUBLIC "BLAS_INT_T=${BLAS_INT_T}" )
endif()

#-------------------------------------------------------------------------------
# Include directory for all targets
include_directories( "${PROJECT_SOURCE_DIR}/include" )

#-------------------------------------------------------------------------------
# Docs
add_subdirectory(docs)

#-------------------------------------------------------------------------------
# Config
add_subdirectory(config)

#-------------------------------------------------------------------------------
# C and Fortran wrappers
add_subdirectory(src)

#-------------------------------------------------------------------------------
# Examples
if( BUILD_EXAMPLES )
  add_subdirectory(examples)
endif()

#-------------------------------------------------------------------------------
# Include tests
include(CTest)
if( BUILD_TESTING )
  add_subdirectory(test)
endif()