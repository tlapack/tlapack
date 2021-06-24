# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Options
option( TEST_MPFR "Use mpreal from MPFR C++ library (http://www.holoborodko.com/pavel/mpfr/) for testing"  OFF )

#-------------------------------------------------------------------------------
# Search for MPFR library if it is needed
if( TEST_MPFR )
  find_package( GMP  4.2.1 REQUIRED )
  find_package( MPFR 2.3.1 REQUIRED )
  find_path( MPREALPATH
    NAMES mpreal.h
    PATHS ${MPFR_INCLUDES} ${GMP_INCLUDES} )
  if( MPREALPATH-NOTFOUND )
    message( FATAL_ERROR "MPFR C++ (MPREAL) not found. Try installing mpreal.h from
    (1) https://github.com/advanpix/mpreal,
 or (2) http://www.holoborodko.com/pavel/mpfr." )
  endif()
endif()

#-------------------------------------------------------------------------------
# Load Catch2
find_package( Catch2 QUIET ) # Try to load Catch2 from the system
if( NOT Catch2_FOUND )

  if  ( EXISTS "${Catch2_DIR}/include/catch.hpp" )

    add_subdirectory( ${Catch2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Catch2 )
    message( STATUS "Using Catch2 from ${Catch2_DIR}" )

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_DIR}/extras;${Catch2_DIR}/contrib" )

  elseif( EXISTS "$ENV{Catch2_DIR}/include/catch.hpp" )

    get_property( docString CACHE Catch2_DIR PROPERTY HELPSTRING )
    set( Catch2_DIR $ENV{Catch2_DIR} CACHE STRING "${docString}" FORCE )
    add_subdirectory( ${Catch2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Catch2 )
    message( STATUS "Using Catch2 from ${Catch2_DIR}" )

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_DIR}/extras;${Catch2_DIR}/contrib" )

  elseif( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14" )

    message( STATUS "Catch2 not found. Trying to fetch from https://github.com/catchorg/Catch2.git. "
                    "It may take a while." )

    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v2.13.1 )

    FetchContent_MakeAvailable(Catch2)

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_SOURCE_DIR}/extras;${Catch2_SOURCE_DIR}/contrib" )

    # Test if the fetch was successful
    if( EXISTS "${Catch2_SOURCE_DIR}/include/catch.hpp" )
      message( STATUS "Using Catch2 from https://github.com/catchorg/Catch2.git." )
    else()
      message( FATAL_ERROR "Failed in fetching Catch2 from https://github.com/catchorg/Catch2.git." )
    endif()

    # Hide Catch2_DIR, CATCH_ and FETCHCONTENT_ options
    mark_as_advanced( FORCE Catch2_DIR )
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
      if( "${_variableName}" MATCHES "^CATCH_" OR
          "${_variableName}" MATCHES "^FETCHCONTENT_" )
        mark_as_advanced( FORCE ${_variableName} )
      endif()
    endforeach()

  else()
    message( FATAL_ERROR "Catch2 not found. Set \"Catch2_DIR\" to a directory containing either
    (1) \"include/catch.hpp\",
 or (2) one of the files: Catch2Config.cmake, catch2-config.cmake." )
  endif()

endif( NOT Catch2_FOUND )

add_subdirectory(blas)