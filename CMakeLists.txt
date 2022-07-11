# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
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

add_library( tblas INTERFACE )
target_include_directories( tblas
  INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include> )

add_library( tlapack INTERFACE )
target_link_libraries( tlapack INTERFACE tblas )

#-------------------------------------------------------------------------------
# Options

# Includes CMAKE_DEPENDENT_OPTION
include(CMakeDependentOption)

# LAPACK++ interface
option( USE_LAPACKPP_WRAPPERS "Use LAPACK++ wrappers to link with optimized BLAS and LAPACK libraries" OFF )

cmake_dependent_option( BUILD_BLASPP_TESTS   "Build BLAS++ tests. Please inform blaspp_TEST_DIR if BUILD_BLASPP_TESTS=ON"
  OFF "BUILD_TESTING" # Default value when condition is true
  OFF # Value when condition is false 
)

cmake_dependent_option( BUILD_LAPACKPP_TESTS "Build LAPACK++ tests. Please inform lapackpp_TEST_DIR if BUILD_LAPACKPP_TESTS=ON"
  OFF "BUILD_TESTING" # Default value when condition is true
  OFF # Value when condition is false
)

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

# Tests
option( TLAPACK_BUILD_SINGLE_TESTER "Build one additional executable that contains all tests" OFF  )

# Wrappers to <T>BLAS and <T>LAPACK
option( C_WRAPPERS       "Build and install C wrappers"               OFF )
option( CBLAS_WRAPPERS   "Build and install CBLAS wrappers to <T>BLAS" OFF )
option( Fortran_WRAPPERS "Build and install Fortran wrappers"         OFF )

# Use of mdspan instead of the <T>LAPACK legacy arrays
option( TLAPACK_USE_MDSPAN "Use mdspan for the legacy wrappers" OFF )
mark_as_advanced( FORCE TLAPACK_USE_MDSPAN )

# Enable disable error checks
option( TLAPACK_NDEBUG "Disable all error checks from <T>LAPACK" OFF )

cmake_dependent_option(
  TLAPACK_CHECK_INPUT "<T>LAPACK routines check if the input parameters are illegal"
  ON "NOT TLAPACK_NDEBUG" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_ENABLE_NANCHECK "Enable NaN checks for the <T>LAPACK routines"
  OFF "NOT TLAPACK_NDEBUG" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_DEFAULT_NANCHECK "By default, <T>LAPACK routines check for NaNs as specified in their documentation"
  ON "NOT TLAPACK_NDEBUG; TLAPACK_ENABLE_NANCHECK" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_ENABLE_INFCHECK "Enable Inf checks for the <T>LAPACK routines"
  OFF "NOT TLAPACK_NDEBUG" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_DEFAULT_INFCHECK "By default, <T>LAPACK routines check for Infs as specified in their documentation"
  ON "NOT TLAPACK_NDEBUG; TLAPACK_ENABLE_INFCHECK" # Default value when condition is true
  OFF # Value when condition is false
)

# Configure compiler flags for Error Checking
if( TLAPACK_NDEBUG )
  target_compile_definitions( tblas INTERFACE TLAPACK_NDEBUG )
else()
  if( TLAPACK_CHECK_INPUT )
    target_compile_definitions( tblas INTERFACE TLAPACK_CHECK_INPUT )
  endif()
  if( TLAPACK_ENABLE_NANCHECK )
    target_compile_definitions( tblas INTERFACE TLAPACK_ENABLE_NANCHECK )
    if( TLAPACK_DEFAULT_NANCHECK )
      target_compile_definitions( tblas INTERFACE TLAPACK_DEFAULT_NANCHECK=1 )
    else()
      target_compile_definitions( tblas INTERFACE TLAPACK_DEFAULT_NANCHECK=0 )
    endif()
  endif()
  if( TLAPACK_ENABLE_INFCHECK )
    target_compile_definitions( tblas INTERFACE TLAPACK_ENABLE_INFCHECK )
    if( TLAPACK_DEFAULT_INFCHECK )
      target_compile_definitions( tblas INTERFACE TLAPACK_DEFAULT_INFCHECK=1 )
    else()
      target_compile_definitions( tblas INTERFACE TLAPACK_DEFAULT_INFCHECK=0 )
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#-------------------------------------------------------------------------------
# Search for LAPACK++ library if it is needed
if( USE_LAPACKPP_WRAPPERS )
  find_package( lapackpp REQUIRED )
  target_compile_definitions( tblas INTERFACE USE_LAPACKPP_WRAPPERS )
  target_link_libraries( tblas INTERFACE lapackpp )
endif()

#-------------------------------------------------------------------------------
# Load mdspan if needed
if( TLAPACK_USE_MDSPAN )
  include( "${TLAPACK_SOURCE_DIR}/cmake/FetchPackage.cmake" )
  FetchPackage( "mdspan" "https://github.com/kokkos/mdspan.git" "stable" )
  if( mdspan_FOUND )
    # enforce the C++14 standard
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )
    target_compile_definitions( tblas INTERFACE TLAPACK_USE_MDSPAN )
    target_link_libraries( tblas INTERFACE std::mdspan )
  else()
    message( STATUS "Legacy wrappers will use legacy arrays instead of mdspan" )
  endif()
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

# <T>LAPACK templates 
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp" )

# Modules
install(
  FILES ${installable_mods}
  DESTINATION include )

# Libraries
list( APPEND installable_libs tblas tlapack )
install(
  TARGETS ${installable_libs}
  EXPORT tlapackTargets
  DESTINATION lib
  INCLUDES DESTINATION include )

# Export targets
install(EXPORT tlapackTargets
  FILE tlapackTargets.cmake
  DESTINATION lib/cmake/tlapack )

# Configure installation
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
