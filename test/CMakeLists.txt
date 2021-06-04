# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of T-LAPACK.
# T-LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Load Catch2
find_package( Catch2 QUIET ) # Try to load Catch2 from the system
if( NOT Catch2_FOUND )

  if  ( EXISTS "${Catch2_DIR}/include/catch.hpp" )

    add_subdirectory( ${Catch2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Catch2 )
    message( STATUS "  Using Catch2 from ${Catch2_DIR}" )

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_DIR}/extras;${Catch2_DIR}/contrib" )

  elseif( EXISTS "$ENV{Catch2_DIR}/include/catch.hpp" )

    set( Catch2_DIR $ENV{Catch2_DIR} )
    add_subdirectory( ${Catch2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Catch2 )
    message( STATUS "  Using Catch2 from ${Catch2_DIR}" )

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_DIR}/extras;${Catch2_DIR}/contrib" )

  elseif( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14" )

    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v2.13.1 )

    FetchContent_MakeAvailable(Catch2)

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_SOURCE_DIR}/extras;${Catch2_SOURCE_DIR}/contrib" )

  else()
    message( FATAL_ERROR "  Catch2 not found. Set \"Catch2_DIR\" to a directory containing either
    (1) \"include/catch.hpp\",
 or (2) one of the files: Catch2Config.cmake, catch2-config.cmake." )
  endif()

endif( NOT Catch2_FOUND )

add_subdirectory(blas)