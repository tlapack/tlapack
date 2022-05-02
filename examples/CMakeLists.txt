# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# add the example geqr2
add_subdirectory( geqr2 )

# add the eigenvalues example
add_subdirectory( eigenvalues )

# add the example gemm (Use MPFR library if it is available)
add_subdirectory( gemm )

# add the example access_types
add_subdirectory( access_types )

find_package( Eigen3 QUIET )
if( Eigen3_FOUND )
  # add the example eigen
  add_subdirectory( eigen )
else()
  mark_as_advanced( FORCE Eigen3_DIR )
endif()

find_package( mdspan QUIET )
if( mdspan_FOUND )
  # add the example mdspan
  add_subdirectory( mdspan )
else()
  mark_as_advanced( FORCE mdspan_DIR )
endif()

# add the example gemm using the C wrapper
if( C_WRAPPERS )
  add_subdirectory( cwrapper_gemm )
endif()

# add the example caxpy using the Fortran module
if( Fortran_WRAPPERS )
  add_subdirectory( fortranModule_caxpy )
endif()

# add the example daxpy using the Fortran wrapper
if( Fortran_WRAPPERS )
add_subdirectory( fortranWrapper_ssymm )
endif()

add_custom_target( run-all-examples
COMMAND
  echo "- example_gemm -----------------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/gemm/example_gemm${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_geqr2 ----------------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/geqr2/example_geqr2${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_accessTypes ----------------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/access_types/example_accessTypes${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_cwrapper_gemm --------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/cwrapper_gemm/example_cwrapper_gemm${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_fortranModule_caxpy --" &&
  ${CMAKE_CURRENT_BINARY_DIR}/fortranModule_caxpy/example_fortranModule_caxpy${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_fortranWrapper_saxpy -" &&
  ${CMAKE_CURRENT_BINARY_DIR}/fortranWrapper_ssymm/example_fortranWrapper_ssymm${CMAKE_EXECUTABLE_SUFFIX} )
