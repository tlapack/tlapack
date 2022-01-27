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

# # specify the C++ standard
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

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

set( installable_libs tblas ) # Mind that we don't install tlapack since it is an INTERFACE_LIBRARY

#-------------------------------------------------------------------------------
# Options

# BLAS++ and LAPACK++ interface
option( USE_BLASPP_WRAPPERS   "Use BLAS++ wrappers to link with an optimized BLAS library"     OFF )
option( USE_LAPACKPP_WRAPPERS "Use LAPACK++ wrappers to link with an optimized LAPACK library" OFF )
option( BUILD_BLASPP_TESTS    "Build BLAS++ tests. Not used if BUILD_TESTING is OFF"           OFF )
option( BUILD_LAPACKPP_TESTS  "Build LAPACK++ tests. Not used if BUILD_TESTING is OFF"         OFF )

# TODO: Try to search this path automatically
set( blaspp_TEST_DIR "" CACHE STRING "Directory with the BLAS++ tests" )
if( NOT BUILD_BLASPP_TESTS )
  mark_as_advanced( FORCE blaspp_TEST_DIR )
else()
  mark_as_advanced( CLEAR blaspp_TEST_DIR )
endif()

# TODO: Try to search this path automatically
set( lapackpp_TEST_DIR "" CACHE STRING "Directory with the LAPACK++ tests" )
if( NOT BUILD_LAPACKPP_TESTS )
  mark_as_advanced( FORCE lapackpp_TEST_DIR )
else()
  mark_as_advanced( CLEAR lapackpp_TEST_DIR )
endif()

# Examples
option( BUILD_EXAMPLES "Build examples" ON  )

# Wrappers to <T>BLAS and <T>LAPACK
option( C_WRAPPERS       "Build and install C wrappers"               OFF )
option( CBLAS_WRAPPERS   "Build and install CBLAS wrappers to <T>BLAS" OFF )
option( Fortran_WRAPPERS "Build and install Fortran wrappers"         OFF )

#-------------------------------------------------------------------------------
# Modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#-------------------------------------------------------------------------------
# Search for BLAS++ library if it is needed
if( USE_BLASPP_WRAPPERS )
  find_package( blaspp REQUIRED )
  target_compile_definitions( tblas PUBLIC USE_BLASPP_WRAPPERS )
  target_link_libraries( tblas PUBLIC blaspp )
endif()

#-------------------------------------------------------------------------------
# Search for LAPACK++ library if it is needed
if( USE_LAPACKPP_WRAPPERS )
  find_package( lapackpp REQUIRED )
  target_compile_definitions( tlapack INTERFACE USE_LAPACKPP_WRAPPERS )
  target_link_libraries( tlapack INTERFACE lapackpp )
endif()

#-------------------------------------------------------------------------------
# Load mdspan
include( "${TLAPACK_SOURCE_DIR}/cmake/FetchPackage.cmake" )
FetchPackage( "mdspan" "https://github.com/kokkos/mdspan.git" "stable" )
target_link_libraries( tblas PUBLIC std::mdspan )

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
  if( Fortran_WRAPPERS )
    set( CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include )
    enable_language( Fortran )
  endif()
  add_subdirectory(src)
endif()

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
  FILES ${installable_mods}
  DESTINATION include )

install(
  FILES
    "${PROJECT_SOURCE_DIR}/include/tblas.hpp"
    "${PROJECT_SOURCE_DIR}/include/tlapack.hpp"
  DESTINATION include )

# <T>LAPACK templates 
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp" )
install( FILES ${PROJECT_SOURCE_DIR}/include/slate_api/lapack/config_lapackppwrappers.h
          DESTINATION include/slate_api/lapack )

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
