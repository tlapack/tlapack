# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

find_package( blaspp REQUIRED )
find_package( LAPACK REQUIRED )

if( EXISTS "${blaspp_TEST_DIR}/test.cc" )

  FetchPackage( "testsweeper" "https://bitbucket.org/icl/testsweeper" "2021.04.00" )

  set( blaspp_tester )
  add_executable(
    blaspp_tester
    test.cpp
    ${blaspp_TEST_DIR}/test_asum.cc
    ${blaspp_TEST_DIR}/test_axpy.cc
    ${blaspp_TEST_DIR}/test_copy.cc
    ${blaspp_TEST_DIR}/test_dot.cc
    ${blaspp_TEST_DIR}/test_dotu.cc
    ${blaspp_TEST_DIR}/test_error.cc
    ${blaspp_TEST_DIR}/test_gemm.cc
    ${blaspp_TEST_DIR}/test_gemv.cc
    ${blaspp_TEST_DIR}/test_ger.cc
    ${blaspp_TEST_DIR}/test_geru.cc
    ${blaspp_TEST_DIR}/test_hemm.cc
    ${blaspp_TEST_DIR}/test_hemv.cc
    ${blaspp_TEST_DIR}/test_her.cc
    ${blaspp_TEST_DIR}/test_her2.cc
    ${blaspp_TEST_DIR}/test_her2k.cc
    ${blaspp_TEST_DIR}/test_herk.cc
    ${blaspp_TEST_DIR}/test_iamax.cc
    ${blaspp_TEST_DIR}/test_max.cc
    ${blaspp_TEST_DIR}/test_nrm2.cc
    ${blaspp_TEST_DIR}/test_rot.cc
    ${blaspp_TEST_DIR}/test_rotg.cc
    ${blaspp_TEST_DIR}/test_rotm.cc
    ${blaspp_TEST_DIR}/test_rotmg.cc
    ${blaspp_TEST_DIR}/test_scal.cc
    ${blaspp_TEST_DIR}/test_swap.cc
    ${blaspp_TEST_DIR}/test_symm.cc
    ${blaspp_TEST_DIR}/test_symv.cc
    ${blaspp_TEST_DIR}/test_syr.cc
    ${blaspp_TEST_DIR}/test_syr2.cc
    ${blaspp_TEST_DIR}/test_syr2k.cc
    ${blaspp_TEST_DIR}/test_syrk.cc
    ${blaspp_TEST_DIR}/test_trmm.cc
    ${blaspp_TEST_DIR}/test_trmv.cc
    ${blaspp_TEST_DIR}/test_trsm.cc
    ${blaspp_TEST_DIR}/test_trsv.cc
    ${blaspp_TEST_DIR}/cblas_wrappers.cc
    ${blaspp_TEST_DIR}/lapack_wrappers.cc
  )

  target_link_libraries(
    blaspp_tester
    testsweeper
    tblas
  )
  target_link_libraries(
    blaspp_tester
    ${LAPACK_LIBRARIES}
    ${blaspp_libraries}
  )

  target_include_directories( blaspp_tester PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${blaspp_TEST_DIR} )
  target_compile_definitions( blaspp_tester PRIVATE ${blaspp_defines} )

  # Copy run_tests script to build directory.
  add_custom_command(
    TARGET blaspp_tester POST_BUILD
    COMMAND
      cp ${CMAKE_CURRENT_SOURCE_DIR}/run_blaspp_tester.py
         ${CMAKE_CURRENT_BINARY_DIR}/run_blaspp_tester.py
  )

  enable_testing()
  add_test( NAME blaspp_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_blaspp_tester.py --quick )

endif()
