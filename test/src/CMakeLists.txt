# Copyright (c) 2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

# Configurations
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/../include" )
link_libraries( Catch2::Catch2 tlapack )
add_definitions( -DCATCH_CONFIG_MAIN ) # Defines that each tester is a single executable
if( TEST_MPFR )
  include_directories( ${MPFR_INCLUDES} ${GMP_INCLUDES} )
  link_libraries( ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
endif()

# Set the directory where to put testers
set( TLAPACK_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test" )

# Testers
add_executable( test_blocked_francis test_blocked_francis.cpp )
add_executable( test_lasy2 test_lasy2.cpp )
add_executable( test_schur_move test_schur_move.cpp )
add_executable( test_transpose test_transpose.cpp )
add_executable( test_unmhr test_unmhr.cpp )
add_executable( test_gehrd test_gehrd.cpp )
add_executable( test_optBLAS test_optBLAS.cpp )
add_executable( test_schur_swap test_schur_swap.cpp )
add_executable( test_unblocked_francis test_unblocked_francis.cpp )
add_executable( test_utils test_utils.cpp )

# Reset the output directory
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${TLAPACK_RUNTIME_OUTPUT_DIRECTORY}" )
  
# Add tests to CTest
if( NOT TLAPACK_BUILD_SINGLE_TESTER )
  include(Catch)
  
  get_property( DIRECTORY_BUILDSYSTEM_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS )
  foreach(target ${DIRECTORY_BUILDSYSTEM_TARGETS})
    catch_discover_tests( ${target} )
  endforeach()
endif()
