# Copyright (c) 2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/../include" )
link_libraries( Catch2::Catch2 tlapack )
if( TEST_MPFR )
  include_directories( ${MPFR_INCLUDES} ${GMP_INCLUDES} )
  link_libraries( ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
endif()

add_executable( test_blocked_francis test_blocked_francis.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_lasy2 test_lasy2.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_schur_move test_schur_move.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_transpose test_transpose.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_unmhr test_unmhr.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_gehrd test_gehrd.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_optBLAS test_optBLAS.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_schur_swap test_schur_swap.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_unblocked_francis test_unblocked_francis.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )
add_executable( test_utils test_utils.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../tests_main.cpp" )

set_target_properties( 
  test_blocked_francis 
  test_lasy2 
  test_schur_move 
  test_transpose 
  test_unmhr 
  test_gehrd 
  test_optBLAS 
  test_schur_swap 
  test_unblocked_francis
  test_utils
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test" )
  
# Add tests to CTest
if( NOT TLAPACK_BUILD_SINGLE_TESTER )
  include(Catch)
  catch_discover_tests(test_blocked_francis )
  catch_discover_tests(test_lasy2 )
  catch_discover_tests(test_schur_move )
  catch_discover_tests(test_transpose )
  catch_discover_tests(test_unmhr )
  catch_discover_tests(test_gehrd )
  catch_discover_tests(test_optBLAS )
  catch_discover_tests(test_schur_swap )
  catch_discover_tests(test_unblocked_francis)
  catch_discover_tests(test_utils)
endif()
