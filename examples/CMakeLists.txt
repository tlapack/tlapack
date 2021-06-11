# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# add the example version
add_executable(example_version version.cpp)
target_include_directories(example_version PRIVATE "${PROJECT_BINARY_DIR}/config")

# add the example gemm
add_executable(example_gemm example_gemm.cpp)
target_link_libraries( example_gemm PRIVATE tblas )

# add the example gemm using the C wrapper
if( C_WRAPPERS )
  add_executable(example_cwrapper_gemm example_cwrapper_gemm.c)
  target_link_libraries( example_cwrapper_gemm PRIVATE blas )
endif()

# add the example saxpy using the Fortran module
if( Fortran_WRAPPERS )
  add_executable( example_fortranModule_saxpy
    example_fortranModule_saxpy.f90 "${PROJECT_SOURCE_DIR}/include/blas.f90" )
  target_link_libraries( example_fortranModule_saxpy PRIVATE blas )
endif()

# # add the example daxpy using the Fortran interface
# add_executable( example_fortranInterface_daxpy example_fortranInterface_daxpy.f90 )
# target_link_libraries( example_fortranInterface_daxpy PRIVATE blas )

# add the example daxpy using the Fortran wrapper
if( Fortran_WRAPPERS )
  add_executable( example_fortranWrapper_saxpy example_fortranWrapper_saxpy.f90 )
  target_link_libraries( example_fortranWrapper_saxpy PRIVATE fortranwrappers_tblas )
endif()

# add the example geqrf
add_executable(example_geqr2 example_geqr2.cpp)
target_link_libraries( example_geqr2 PRIVATE tlapack )

add_custom_target( run-all-examples
COMMAND
  echo "- example_version --------------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/example_version${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_gemm -----------------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/example_gemm${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_cwrapper_gemm --------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/example_cwrapper_gemm${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_fortranModule_saxpy --" &&
  ${CMAKE_CURRENT_BINARY_DIR}/example_fortranModule_saxpy${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_fortranWrapper_saxpy -" &&
  ${CMAKE_CURRENT_BINARY_DIR}/example_fortranWrapper_saxpy${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_geqr2 ----------------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/example_geqr2${CMAKE_EXECUTABLE_SUFFIX} )