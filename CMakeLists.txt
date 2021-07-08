# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.3)
# VERSION 3.3: IN_LIST for if() operator

#-------------------------------------------------------------------------------
# Read project version
file(READ version.txt version_txt)

#-------------------------------------------------------------------------------
# <T>LAPACK project
project(TLAPACK
  VERSION ${version_txt}
  LANGUAGES CXX)
message(STATUS "<T>LAPACK version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# <T>LAPACK and <T>BLAS template libraries

add_library( tblas src/xerbla.cpp )
target_include_directories( tblas
  PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include> )

add_library( tlapack INTERFACE )
target_link_libraries( tlapack INTERFACE tblas )

set( installable_libs  tlapack tblas )
set( installable_files
  "${PROJECT_SOURCE_DIR}/include/tblas.hpp"
  "${PROJECT_SOURCE_DIR}/include/tlapack.hpp" )

set( TBLAS_DEFINES "" )

#-------------------------------------------------------------------------------
# Options

# BLAS++ and LAPACK++ interface
option( USE_BLASPP_WRAPPERS   "Use BLAS++ wrappers to link with an optimized BLAS library"     OFF )
option( USE_LAPACKPP_WRAPPERS "Use LAPACK++ wrappers to link with an optimized LAPACK library" OFF )

# Examples
option( BUILD_EXAMPLES "Build examples" ON  )

# Wrappers to <T>BLAS and <T>LAPACK
option( C_WRAPPERS       "Build and install C wrappers"               OFF )
option( CBLAS_WRAPPERS   "Build and install CBLAS wrappers to <T>BLAS" OFF )
option( Fortran_WRAPPERS "Build and install Fortran wrappers"         OFF )

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
  "Integer type for all variables except array sizes and matrix leading dimension" )
set_property( CACHE BLAS_INT_T
  PROPERTY STRINGS ${blas_intt_list} )

set( BLAS_SIZE_T size_t CACHE STRING
  "Integer type for array sizes, array indexes, and matrix leading dimension" )
set_property( CACHE BLAS_SIZE_T
  PROPERTY STRINGS ${blas_intt_list} size_t uint8_t uint16_t uint32_t uint64_t )

#-------------------------------------------------------------------------------
# Modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#-------------------------------------------------------------------------------
# Search for BLAS++ library if it is needed

if( USE_BLASPP_WRAPPERS )
  find_package( blaspp )
  if(blaspp_FOUND)
    message( STATUS "  BLAS++ found in ${blaspp_DIR}" )
  else()
    message( FATAL_ERROR "  BLAS++ not found." )
  endif()
  target_compile_definitions( tblas PUBLIC USE_BLASPP_WRAPPERS )
  target_link_libraries( tblas PUBLIC blaspp )
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
# Options for BLAS_SIZE_T

if( USE_BLASPP_WRAPPERS OR USE_LAPACKPP_WRAPPERS )
  mark_as_advanced( FORCE BLAS_SIZE_T )
  get_property( docString CACHE BLAS_SIZE_T PROPERTY HELPSTRING )
  set( BLAS_SIZE_T int64_t CACHE STRING "${docString}" FORCE )
else()
  mark_as_advanced( CLEAR BLAS_SIZE_T )
endif()

if( NOT BLAS_SIZE_T STREQUAL "" )
  string( APPEND TBLAS_DEFINES "#define BLAS_SIZE_T ${BLAS_SIZE_T}\n" )
endif()

#-------------------------------------------------------------------------------
# Options for BLAS_INT_T
  
if( USE_BLASPP_WRAPPERS OR USE_LAPACKPP_WRAPPERS )
  mark_as_advanced( FORCE BLAS_INT_T )
  get_property( docString CACHE BLAS_INT_T PROPERTY HELPSTRING )
  set( BLAS_INT_T int64_t CACHE STRING "${docString}" FORCE )
else()
  mark_as_advanced( CLEAR BLAS_INT_T )
endif()

if( NOT BLAS_INT_T STREQUAL "" )
  string( APPEND TBLAS_DEFINES "#define BLAS_INT_T ${BLAS_INT_T}\n" )
endif()

#-------------------------------------------------------------------------------
# Docs
add_subdirectory(docs)

#-------------------------------------------------------------------------------
# Config
add_subdirectory(config)

#-------------------------------------------------------------------------------
# C and Fortran wrappers

if( C_WRAPPERS OR CBLAS_WRAPPERS OR Fortran_WRAPPERS )
  enable_language( C )
endif()
if( Fortran_WRAPPERS )
  enable_language( Fortran )
endif()

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

#-------------------------------------------------------------------------------
# Common configurations

configure_file(
  ${PROJECT_SOURCE_DIR}/include/blas/defines.h.in
  ${CMAKE_BINARY_DIR}/include/blas/defines.h @ONLY )

set_target_properties( ${installable_libs}
PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )

#-------------------------------------------------------------------------------
# Install

install(
  TARGETS ${installable_libs}
  DESTINATION lib
  EXPORT tlapackTargets )
install(
  FILES ${installable_files}
  DESTINATION include )

# <T>LAPACK templates 
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/lapack/
  DESTINATION include/lapack
  FILES_MATCHING PATTERN "*.hpp" )
install( FILES ${PROJECT_SOURCE_DIR}/include/lapack/config.h DESTINATION include/lapack )

# <T>BLAS templates 
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/blas/
  DESTINATION include/blas
  FILES_MATCHING PATTERN "*.hpp" )
install(
  FILES ${CMAKE_BINARY_DIR}/include/blas/defines.h ${PROJECT_SOURCE_DIR}/include/blas/config.h
  DESTINATION include/blas )

# Export targets
install(EXPORT tlapackTargets
  FILE tlapackTargets.cmake
  DESTINATION lib/cmake/tlapack )

include(CMakePackageConfigHelpers)
# generate the config file that includes the exports
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  tlapackConfig.cmake
  INSTALL_DESTINATION lib/cmake/tlapack )
# generate the version file for the config file
write_basic_package_version_file(
  tlapackConfigVersion.cmake
  VERSION "${TLAPACK_VERSION}"
  COMPATIBILITY AnyNewerVersion )

# Install the configuration file
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/tlapackConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/tlapackConfigVersion.cmake
  DESTINATION lib/cmake/tlapack )

# Export targets to the build directory
export(
  EXPORT tlapackTargets
  FILE tlapackTargets.cmake )