# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Load testBLAS
find_package( testBLAS QUIET ) # Try to load testBLAS from the system
if( NOT testBLAS_FOUND )

  if  ( EXISTS "${testBLAS_DIR}" )

    add_subdirectory( ${testBLAS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/testBLAS )
    message( STATUS "Using testBLAS from ${testBLAS_DIR}" )

  elseif( EXISTS "$ENV{testBLAS_DIR}" )

    get_property( docString CACHE testBLAS_DIR PROPERTY HELPSTRING )
    set( testBLAS_DIR $ENV{testBLAS_DIR} CACHE STRING "${docString}" FORCE )

    add_subdirectory( ${testBLAS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/testBLAS )
    message( STATUS "Using testBLAS from ${testBLAS_DIR}" )

  elseif( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14" )

    message( STATUS "testBLAS not found. Trying to fetch from https://github.com/tlapack/testBLAS.git. "
                    "It may take a while." )

    include(FetchContent)
    FetchContent_Declare(
      testBLAS
      GIT_REPOSITORY https://github.com/tlapack/testBLAS.git )

    FetchContent_MakeAvailable(testBLAS)

    # Test if the fetch was successful
    if( EXISTS "${testBLAS_SOURCE_DIR}" )
      message( STATUS "Using testBLAS from https://github.com/tlapack/testBLAS.git." )
    else()
      message( FATAL_ERROR "Failed in fetching testBLAS from https://github.com/tlapack/testBLAS.git." )
    endif()

    # Hide testBLAS_DIR and FETCHCONTENT_ options
    mark_as_advanced( FORCE testBLAS_DIR )
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
      if( "${_variableName}" MATCHES "^FETCHCONTENT_" )
        mark_as_advanced( FORCE ${_variableName} )
      endif()
    endforeach()

  else()
    message( FATAL_ERROR "testBLAS not found. Set \"testBLAS_DIR\" to a directory containing "
                         "one of the files: testBLASConfig.cmake, testBLAS-config.cmake; or "
                         "containing the root of the project testBLAS." )
  endif()

endif( NOT testBLAS_FOUND )