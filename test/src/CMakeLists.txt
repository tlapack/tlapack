# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

# Configurations
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include")
link_libraries(Catch2::Catch2WithMain tlapack)

if(TLAPACK_TEST_EIGEN)
  find_package(Eigen3 REQUIRED)
  link_libraries(Eigen3::Eigen)
  add_compile_definitions(TLAPACK_TEST_EIGEN)
endif()

if(TLAPACK_TEST_MDSPAN)
  find_package(mdspan REQUIRED)
  link_libraries(std::mdspan)
  add_compile_definitions(TLAPACK_TEST_MDSPAN)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(TLAPACK_TEST_MPFR)
  find_package(MPFR 2.3.1 REQUIRED)
  find_package(GMP 4.2.1 REQUIRED)
  include_directories(${MPFR_INCLUDES} ${GMP_INCLUDES})
  link_libraries(${MPFR_LIBRARIES} ${GMP_LIBRARIES})
  add_compile_definitions(TLAPACK_TEST_MPFR)
endif()

# Set the directory where to put testers
set(TLAPACK_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test")

# Testers
add_executable(test_blocked_francis test_blocked_francis.cpp)
add_executable(test_lasy2 test_lasy2.cpp)
add_executable(test_schur_move test_schur_move.cpp)
add_executable(test_transpose test_transpose.cpp)
add_executable(test_unmhr test_unmhr.cpp)
add_executable(test_gehrd test_gehrd.cpp)
add_executable(test_optBLAS test_optBLAS.cpp)
add_executable(test_schur_swap test_schur_swap.cpp)
add_executable(test_unblocked_francis test_unblocked_francis.cpp)
add_executable(test_utils test_utils.cpp)
add_executable(test_gelqf test_gelqf.cpp)
add_executable(test_gelqt test_gelqt.cpp)
add_executable(test_gelq2 test_gelq2.cpp)
add_executable(test_geqr2 test_geqr2.cpp)
add_executable(test_geqrf test_geqrf.cpp)
add_executable(test_gerqf test_gerqf.cpp)
add_executable(test_geql2 test_geql2.cpp)
add_executable(test_geqlf test_geqlf.cpp)
add_executable(test_gerq2 test_gerq2.cpp)
add_executable(test_trtri test_trtri.cpp)
add_executable(test_gebd2 test_gebd2.cpp)
add_executable(test_lu_mult test_lu_mult.cpp)
add_executable(test_getrf test_getrf.cpp)
add_executable(test_getri test_getri.cpp)
add_executable(test_ul_mult test_ul_mult.cpp)
add_executable(test_unmr2 test_unmr2.cpp)
add_executable(test_unm2r test_unm2r.cpp)
add_executable(test_unmqr test_unmqr.cpp)
add_executable(test_unml2 test_unml2.cpp)
add_executable(test_unm2l test_unm2l.cpp)
add_executable(test_lauum test_lauum.cpp)
add_executable(test_potrf test_potrf.cpp)
add_executable(test_larf test_larf.cpp)

if(TLAPACK_TEST_EIGEN)
  add_executable(test_eigenplugin test_eigenplugin.cpp)
endif()

if(TLAPACK_TEST_MDSPAN)
  add_executable(test_mdspanplugin test_mdspanplugin.cpp)
endif()

# Reset the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${TLAPACK_RUNTIME_OUTPUT_DIRECTORY}")

# Add tests to CTest
include(Catch)

get_property(DIRECTORY_BUILDSYSTEM_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)

foreach(target ${DIRECTORY_BUILDSYSTEM_TARGETS})
  catch_discover_tests(${target})
endforeach()
