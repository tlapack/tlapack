## @file SetUpConfigurations.cmake
## Adapted from @see https://stackoverflow.com/questions/31546278/where-to-set-cmake-configuration-types-in-a-project-with-subprojects
#
# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

if(NOT SET_UP_CONFIGURATIONS_DONE)
  set(SET_UP_CONFIGURATIONS_DONE TRUE)

  # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
  # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
  get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(isMultiConfig)
    get_property( docString CACHE CMAKE_CONFIGURATION_TYPES PROPERTY HELPSTRING )
    set(CMAKE_CONFIGURATION_TYPES
      "Debug;Release;MinSizeRel;RelWithDebInfo;Coverage" CACHE STRING "${docString}" FORCE) 
  else()
    if(NOT CMAKE_BUILD_TYPE)
      message( STATUS "Setting build type to 'Release' as none was specified." )
      set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
    endif()
    # set the valid options for cmake-gui drop-down list
    set_property( CACHE CMAKE_BUILD_TYPE
      PROPERTY STRINGS
        "Debug;Release;MinSizeRel;RelWithDebInfo;Coverage" )
  endif()

  # now set up the Coverage configuration
  set( CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG}" )
  set( CMAKE_Fortran_FLAGS_COVERAGE "${CMAKE_Fortran_FLAGS_DEBUG}" )
  set( CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG}" )
  set( CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )
  set( CMAKE_STATIC_LINKER_FLAGS_COVERAGE "${CMAKE_STATIC_LINKER_FLAGS_DEBUG}" )
  set( CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )

endif()