# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

find_package( blaspp REQUIRED )
find_package( LAPACK REQUIRED )
find_package( lapackpp REQUIRED )

if( EXISTS "${lapackpp_TEST_DIR}/test.cc" )

  FetchPackage( "testsweeper" "https://bitbucket.org/icl/testsweeper" "2021.04.00" )

  set( lapackpp_tester )
  add_executable(
    lapackpp_tester
    test.cpp
    ${lapackpp_TEST_DIR}/cblas_wrappers.cc
    ${lapackpp_TEST_DIR}/matrix_generator.cc
    ${lapackpp_TEST_DIR}/matrix_params.cc
    # ${lapackpp_TEST_DIR}/test_gbcon.cc
    # ${lapackpp_TEST_DIR}/test_gbequ.cc
    # ${lapackpp_TEST_DIR}/test_gbrfs.cc
    # ${lapackpp_TEST_DIR}/test_gbsv.cc
    # ${lapackpp_TEST_DIR}/test_gbtrf.cc
    # ${lapackpp_TEST_DIR}/test_gbtrs.cc
    # ${lapackpp_TEST_DIR}/test_gecon.cc
    # ${lapackpp_TEST_DIR}/test_geequ.cc
    # ${lapackpp_TEST_DIR}/test_geev.cc
    # ${lapackpp_TEST_DIR}/test_gehrd.cc
    # ${lapackpp_TEST_DIR}/test_gelqf.cc
    # ${lapackpp_TEST_DIR}/test_gels.cc
    # ${lapackpp_TEST_DIR}/test_gelsd.cc
    # ${lapackpp_TEST_DIR}/test_gelss.cc
    # ${lapackpp_TEST_DIR}/test_gelsy.cc
    # ${lapackpp_TEST_DIR}/test_geqlf.cc
    # ${lapackpp_TEST_DIR}/test_geqrf.cc
    # ${lapackpp_TEST_DIR}/test_gerfs.cc
    # ${lapackpp_TEST_DIR}/test_gerqf.cc
    # ${lapackpp_TEST_DIR}/test_gesdd.cc
    # ${lapackpp_TEST_DIR}/test_gesv.cc
    # ${lapackpp_TEST_DIR}/test_gesvd.cc
    # ${lapackpp_TEST_DIR}/test_gesvdx.cc
    # ${lapackpp_TEST_DIR}/test_gesvx.cc
    # ${lapackpp_TEST_DIR}/test_getrf.cc
    # ${lapackpp_TEST_DIR}/test_getri.cc
    # ${lapackpp_TEST_DIR}/test_getrs.cc
    # ${lapackpp_TEST_DIR}/test_getsls.cc
    # ${lapackpp_TEST_DIR}/test_ggev.cc
    # ${lapackpp_TEST_DIR}/test_ggglm.cc
    # ${lapackpp_TEST_DIR}/test_gglse.cc
    # ${lapackpp_TEST_DIR}/test_ggqrf.cc
    # ${lapackpp_TEST_DIR}/test_ggrqf.cc
    # ${lapackpp_TEST_DIR}/test_gtcon.cc
    # ${lapackpp_TEST_DIR}/test_gtrfs.cc
    # ${lapackpp_TEST_DIR}/test_gtsv.cc
    # ${lapackpp_TEST_DIR}/test_gttrf.cc
    # ${lapackpp_TEST_DIR}/test_gttrs.cc
    # ${lapackpp_TEST_DIR}/test_hbev.cc
    # ${lapackpp_TEST_DIR}/test_hbevd.cc
    # ${lapackpp_TEST_DIR}/test_hbevx.cc
    # ${lapackpp_TEST_DIR}/test_hbgv.cc
    # ${lapackpp_TEST_DIR}/test_hbgvd.cc
    # ${lapackpp_TEST_DIR}/test_hbgvx.cc
    # ${lapackpp_TEST_DIR}/test_hecon.cc
    # ${lapackpp_TEST_DIR}/test_heev.cc
    # ${lapackpp_TEST_DIR}/test_heevd.cc
    # ${lapackpp_TEST_DIR}/test_heevr.cc
    # ${lapackpp_TEST_DIR}/test_heevx.cc
    # ${lapackpp_TEST_DIR}/test_hegst.cc
    # ${lapackpp_TEST_DIR}/test_hegv.cc
    # ${lapackpp_TEST_DIR}/test_hegvd.cc
    # ${lapackpp_TEST_DIR}/test_hegvx.cc
    # ${lapackpp_TEST_DIR}/test_herfs.cc
    # ${lapackpp_TEST_DIR}/test_hesv.cc
    # ${lapackpp_TEST_DIR}/test_hetrd.cc
    # ${lapackpp_TEST_DIR}/test_hetrf.cc
    # ${lapackpp_TEST_DIR}/test_hetri.cc
    # ${lapackpp_TEST_DIR}/test_hetrs.cc
    # ${lapackpp_TEST_DIR}/test_hpcon.cc
    # ${lapackpp_TEST_DIR}/test_hpev.cc
    # ${lapackpp_TEST_DIR}/test_hpevd.cc
    # ${lapackpp_TEST_DIR}/test_hpevx.cc
    # ${lapackpp_TEST_DIR}/test_hpgst.cc
    # ${lapackpp_TEST_DIR}/test_hpgv.cc
    # ${lapackpp_TEST_DIR}/test_hpgvd.cc
    # ${lapackpp_TEST_DIR}/test_hpgvx.cc
    # ${lapackpp_TEST_DIR}/test_hprfs.cc
    # ${lapackpp_TEST_DIR}/test_hpsv.cc
    # ${lapackpp_TEST_DIR}/test_hptrd.cc
    # ${lapackpp_TEST_DIR}/test_hptrf.cc
    # ${lapackpp_TEST_DIR}/test_hptri.cc
    # ${lapackpp_TEST_DIR}/test_hptrs.cc
    ${lapackpp_TEST_DIR}/test_lacpy.cc
    # ${lapackpp_TEST_DIR}/test_langb.cc
    ${lapackpp_TEST_DIR}/test_lange.cc
    # ${lapackpp_TEST_DIR}/test_langt.cc
    # ${lapackpp_TEST_DIR}/test_lanhb.cc
    # ${lapackpp_TEST_DIR}/test_lanhe.cc
    # ${lapackpp_TEST_DIR}/test_lanhp.cc
    # ${lapackpp_TEST_DIR}/test_lanhs.cc
    # ${lapackpp_TEST_DIR}/test_lanst.cc
    # ${lapackpp_TEST_DIR}/test_lanht.cc
    # ${lapackpp_TEST_DIR}/test_lansb.cc
    # ${lapackpp_TEST_DIR}/test_lansp.cc
    ${lapackpp_TEST_DIR}/test_lansy.cc
    # ${lapackpp_TEST_DIR}/test_lantb.cc
    # ${lapackpp_TEST_DIR}/test_lantp.cc
    # ${lapackpp_TEST_DIR}/test_lantr.cc
    ${lapackpp_TEST_DIR}/test_larf.cc
    ${lapackpp_TEST_DIR}/test_larfb.cc
    # ${lapackpp_TEST_DIR}/test_larfg.cc  # Disabled because the forward errors don't make much sense in this case 
    # ${lapackpp_TEST_DIR}/test_larfgp.cc
    ${lapackpp_TEST_DIR}/test_larft.cc
    # ${lapackpp_TEST_DIR}/test_larfx.cc
    # ${lapackpp_TEST_DIR}/test_larfy.cc
    ${lapackpp_TEST_DIR}/test_laset.cc
    # ${lapackpp_TEST_DIR}/test_laswp.cc
    # ${lapackpp_TEST_DIR}/test_pbcon.cc
    # ${lapackpp_TEST_DIR}/test_pbequ.cc
    # ${lapackpp_TEST_DIR}/test_pbrfs.cc
    # ${lapackpp_TEST_DIR}/test_pbsv.cc
    # ${lapackpp_TEST_DIR}/test_pbtrf.cc
    # ${lapackpp_TEST_DIR}/test_pbtrs.cc
    # ${lapackpp_TEST_DIR}/test_pocon.cc
    # ${lapackpp_TEST_DIR}/test_poequ.cc
    # ${lapackpp_TEST_DIR}/test_porfs.cc
    # ${lapackpp_TEST_DIR}/test_posv.cc
    # ${lapackpp_TEST_DIR}/test_potrf.cc
    # ${lapackpp_TEST_DIR}/test_potri.cc
    # ${lapackpp_TEST_DIR}/test_potrs.cc
    # ${lapackpp_TEST_DIR}/test_ppcon.cc
    # ${lapackpp_TEST_DIR}/test_ppequ.cc
    # ${lapackpp_TEST_DIR}/test_pprfs.cc
    # ${lapackpp_TEST_DIR}/test_ppsv.cc
    # ${lapackpp_TEST_DIR}/test_pptrf.cc
    # ${lapackpp_TEST_DIR}/test_pptri.cc
    # ${lapackpp_TEST_DIR}/test_pptrs.cc
    # ${lapackpp_TEST_DIR}/test_ptcon.cc
    # ${lapackpp_TEST_DIR}/test_ptrfs.cc
    # ${lapackpp_TEST_DIR}/test_ptsv.cc
    # ${lapackpp_TEST_DIR}/test_pttrf.cc
    # ${lapackpp_TEST_DIR}/test_pttrs.cc
    # ${lapackpp_TEST_DIR}/test_spcon.cc
    # ${lapackpp_TEST_DIR}/test_sprfs.cc
    # ${lapackpp_TEST_DIR}/test_spsv.cc
    # ${lapackpp_TEST_DIR}/test_sptrf.cc
    # ${lapackpp_TEST_DIR}/test_sptri.cc
    # ${lapackpp_TEST_DIR}/test_sptrs.cc
    # ${lapackpp_TEST_DIR}/test_sycon.cc
    # ${lapackpp_TEST_DIR}/test_syr.cc
    # ${lapackpp_TEST_DIR}/test_syrfs.cc
    # ${lapackpp_TEST_DIR}/test_sysv.cc
    # ${lapackpp_TEST_DIR}/test_sysv_aa.cc
    # ${lapackpp_TEST_DIR}/test_sysv_rk.cc
    # ${lapackpp_TEST_DIR}/test_sysv_rook.cc
    # ${lapackpp_TEST_DIR}/test_sytrf.cc
    # ${lapackpp_TEST_DIR}/test_sytrf_aa.cc
    # ${lapackpp_TEST_DIR}/test_sytrf_rk.cc
    # ${lapackpp_TEST_DIR}/test_sytrf_rook.cc
    # ${lapackpp_TEST_DIR}/test_sytri.cc
    # ${lapackpp_TEST_DIR}/test_sytrs.cc
    # ${lapackpp_TEST_DIR}/test_sytrs_aa.cc
    # ${lapackpp_TEST_DIR}/test_sytrs_rook.cc
    # ${lapackpp_TEST_DIR}/test_unghr.cc
    # ${lapackpp_TEST_DIR}/test_unglq.cc
    # ${lapackpp_TEST_DIR}/test_ungql.cc
    # ${lapackpp_TEST_DIR}/test_ungqr.cc
    # ${lapackpp_TEST_DIR}/test_ungrq.cc
    # ${lapackpp_TEST_DIR}/test_ungtr.cc
    # ${lapackpp_TEST_DIR}/test_unmhr.cc
    # ${lapackpp_TEST_DIR}/test_unmtr.cc
    # ${lapackpp_TEST_DIR}/test_upgtr.cc
    # ${lapackpp_TEST_DIR}/test_upmtr.cc
    # ${lapackpp_TEST_DIR}/test_tplqt.cc
    # ${lapackpp_TEST_DIR}/test_tplqt2.cc
    # ${lapackpp_TEST_DIR}/test_tpmlqt.cc
    # ${lapackpp_TEST_DIR}/test_tpmqrt.cc
    # ${lapackpp_TEST_DIR}/test_tpqrt.cc
    # ${lapackpp_TEST_DIR}/test_tpqrt2.cc
    # ${lapackpp_TEST_DIR}/test_tprfb.cc
    # ${lapackpp_TEST_DIR}/test_symv.cc
    # ${lapackpp_TEST_DIR}/test_larfy.cc
  )

  target_link_libraries(
    lapackpp_tester
    testsweeper
    tlapack
    tblas
  )
  target_link_libraries(
    lapackpp_tester
    ${LAPACK_LIBRARIES}
    "-llapacke"          # TODO: Improve this link.
  )

  target_include_directories( lapackpp_tester PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../blaspp
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${lapackpp_TEST_DIR} )
  target_compile_definitions( lapackpp_tester PRIVATE ${lapackpp_defines} ${blaspp_defines} )

  # Copy run_tests script to build directory.
  add_custom_command(
    TARGET lapackpp_tester POST_BUILD
    COMMAND
      cp ${CMAKE_CURRENT_SOURCE_DIR}/run_lapackpp_tester.py
         ${CMAKE_CURRENT_BINARY_DIR}/run_lapackpp_tester.py
  )

  enable_testing()
  add_test( NAME lapackpp_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_lapackpp_tester.py --quick )

endif()
