# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Search for Catch2
find_package( Catch2 QUIET )
if( NOT Catch2_FOUND )
  find_package( Catch2 REQUIRED
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty" )
  message( STATUS "  Using Catch2 from ${Catch2_DIR}" )
endif()

add_subdirectory(blas)