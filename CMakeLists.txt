# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.7)
# VERSION 3.3: IN_LIST for if() operator
# VERSION 3.7: VERSION_GREATER_EQUAL

#-------------------------------------------------------------------------------
# Read project version
file(READ version.txt version_txt)

#
# We require C++17 features like if constexpr
# If possible, compile with C++20 so we can use concepts
set (CMAKE_CXX_STANDARD_REQUIRED 17)
set (CMAKE_CXX_STANDARD 20)

#-------------------------------------------------------------------------------
# <T>LAPACK project
project(TLAPACK
  VERSION ${version_txt}
  LANGUAGES CXX)
message(STATUS "<T>LAPACK version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# <T>LAPACK template libraries

add_library( tlapack INTERFACE )
target_include_directories( tlapack
  INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include> )

#-------------------------------------------------------------------------------
# Options

# Includes CMAKE_DEPENDENT_OPTION
include(CMakeDependentOption)

# Includes BUILD_TESTING option
include(CTest)

# LAPACK++ wrappers
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
option( TLAPACK_TEST_EIGEN "Add Eigen matrices to the types to test" OFF )
option( TLAPACK_TEST_MDSPAN "Add mdspan matrices to the types to test" OFF )
option( TLAPACK_TEST_MPFR "Add GNU multiprecision type to test" OFF )

# Wrappers to <T>LAPACK
option( BUILD_C_WRAPPERS       "Build and install C wrappers (WIP)" OFF )
option( BUILD_CBLAS_WRAPPERS   "Build and install CBLAS wrappers (WIP)" OFF )
option( BUILD_Fortran_WRAPPERS "Build and install Fortran wrappers (WIP)" OFF )

# Use of mdspan instead of the <T>LAPACK legacy arrays
option( TLAPACK_USE_MDSPAN "Use mdspan for the legacy wrappers" OFF )
mark_as_advanced( FORCE TLAPACK_USE_MDSPAN )

# Enable disable error checks
option( TLAPACK_NDEBUG "Disable all error checks" OFF )

cmake_dependent_option(
  TLAPACK_CHECK_INPUT "Enable checks on input arguments"
  ON "NOT TLAPACK_NDEBUG" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_ENABLE_NANCHECK "Enable check for NaNs as specified in the documentation of each routine"
  OFF "NOT TLAPACK_NDEBUG" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_DEFAULT_NANCHECK "Default behavior of checks for NaNs. Checks can be activated/deactivated at runtime"
  ON "NOT TLAPACK_NDEBUG; TLAPACK_ENABLE_NANCHECK" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_ENABLE_INFCHECK "Enable check for Infs as specified in the documentation of each routine"
  OFF "NOT TLAPACK_NDEBUG" # Default value when condition is true
  OFF # Value when condition is false
)

cmake_dependent_option(
  TLAPACK_DEFAULT_INFCHECK "Default behavior of checks for Infs. Checks can be activated/deactivated at runtime"
  ON "NOT TLAPACK_NDEBUG; TLAPACK_ENABLE_INFCHECK" # Default value when condition is true
  OFF # Value when condition is false
)

# Configure compiler flags for Error Checking
if( TLAPACK_NDEBUG )
  target_compile_definitions( tlapack INTERFACE TLAPACK_NDEBUG )
else()
  if( TLAPACK_CHECK_INPUT )
    target_compile_definitions( tlapack INTERFACE TLAPACK_CHECK_INPUT )
  endif()
  if( TLAPACK_ENABLE_NANCHECK )
    target_compile_definitions( tlapack INTERFACE TLAPACK_ENABLE_NANCHECK )
    if( TLAPACK_DEFAULT_NANCHECK )
      target_compile_definitions( tlapack INTERFACE TLAPACK_DEFAULT_NANCHECK=1 )
    else()
      target_compile_definitions( tlapack INTERFACE TLAPACK_DEFAULT_NANCHECK=0 )
    endif()
  endif()
  if( TLAPACK_ENABLE_INFCHECK )
    target_compile_definitions( tlapack INTERFACE TLAPACK_ENABLE_INFCHECK )
    if( TLAPACK_DEFAULT_INFCHECK )
      target_compile_definitions( tlapack INTERFACE TLAPACK_DEFAULT_INFCHECK=1 )
    else()
      target_compile_definitions( tlapack INTERFACE TLAPACK_DEFAULT_INFCHECK=0 )
    endif()
  endif()
endif()

# Option GIT_SUBMODULE
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

cmake_dependent_option( BUILD_testBLAS_TESTS "Build testBLAS tests"
  ON "BUILD_TESTING" # Default value when condition is true
  OFF # Value when condition is false 
)

#-------------------------------------------------------------------------------
# Modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#-------------------------------------------------------------------------------
# Search for LAPACK++ library if it is needed
if( USE_LAPACKPP_WRAPPERS )
  find_package( lapackpp REQUIRED )
  target_compile_definitions( tlapack INTERFACE USE_LAPACKPP_WRAPPERS )
  target_link_libraries( tlapack INTERFACE lapackpp )
endif()

#-------------------------------------------------------------------------------
# Load mdspan if needed
if( TLAPACK_USE_MDSPAN )
  include( "${TLAPACK_SOURCE_DIR}/cmake/FetchPackage.cmake" )
  FetchPackage( "mdspan" "https://github.com/kokkos/mdspan.git" "stable" )
  if( mdspan_FOUND )
    target_compile_definitions( tlapack INTERFACE TLAPACK_USE_MDSPAN )
    target_link_libraries( tlapack INTERFACE std::mdspan )
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
# Format code
find_package(Git QUIET)
find_package(ClangFormat QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git" AND CLANG_FORMAT_FOUND)
  message(STATUS "ClangFormat found: ${CLANG_FORMAT_EXECUTABLE} (version ${CLANG_FORMAT_VERSION})")
  if( ${CLANG_FORMAT_MAJOR_VERSION} NOT EQUAL 10 )
    message( WARNING "Please use ClangFormat version 10" )
  endif()
  add_custom_target( format-all-hpp-cpp-files
    COMMAND
      ${GIT_EXECUTABLE} ls-tree -r HEAD --name-only |
      grep -E '\(config/|examples/|include/|src/|test/include/|test/src/\).*\\.\(hpp|cpp|h.in|c\)$$' |
      xargs ${CLANG_FORMAT_EXECUTABLE} -i -style=file
      > format-all-hpp-cpp-files.log
      2> format-all-hpp-cpp-files.err
    WORKING_DIRECTORY ${TLAPACK_SOURCE_DIR}
  )
  add_custom_target( check-format-of-hpp-cpp-files
    COMMAND
      ${GIT_EXECUTABLE} ls-tree -r HEAD --name-only |
      grep -E '\(config/|examples/|include/|src/|test/include/|test/src/\).*\\.\(hpp|cpp|h.in|c\)$$' |
      xargs ${CLANG_FORMAT_EXECUTABLE} -i -style=file --dry-run
      > format-all-hpp-cpp-files.log
      2> format-all-hpp-cpp-files.err
    WORKING_DIRECTORY ${TLAPACK_SOURCE_DIR}
  )
endif()

#-------------------------------------------------------------------------------
# C and Fortran wrappers

if( BUILD_C_WRAPPERS OR BUILD_CBLAS_WRAPPERS OR BUILD_Fortran_WRAPPERS )
  enable_language( C )
  if( BUILD_Fortran_WRAPPERS )
    set( CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran )
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
  DESTINATION include/tlapack/fortran )

# Libraries
list( APPEND installable_libs tlapack )
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
