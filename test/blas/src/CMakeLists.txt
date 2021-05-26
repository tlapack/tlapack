# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Test sources
file( GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp" )

#-------------------------------------------------------------------------------
# Test library
add_library( testBLAS ${test_sources} )

target_compile_definitions( testBLAS PUBLIC ${tlapack_defs} )
target_include_directories( testBLAS
    PUBLIC "${PROJECT_SOURCE_DIR}/include" ${tblas_include_dir} )

if( USE_BLASPP )
    target_link_libraries( testBLAS PUBLIC blaspp )
endif()
target_link_libraries( testBLAS PUBLIC Catch2::Catch2 )

set_target_properties( testBLAS
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
target_include_directories( testBLAS INTERFACE "${CMAKE_BINARY_DIR}/lib" )