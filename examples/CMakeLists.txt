# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# add the example geqr2
add_subdirectory( geqr2 )

# add the example gemm (Use MPFR library if it is available)
add_subdirectory( gemm )

# add the example gemm using the C wrapper
if( C_WRAPPERS )
  add_subdirectory( cwrapper_gemm )
endif()

# add the example caxpy using the Fortran module
if( Fortran_WRAPPERS )
  add_subdirectory( fortranModule_caxpy )
endif()

# # add the example daxpy using the Fortran interface
# add_executable( example_fortranInterface_daxpy example_fortranInterface_daxpy.f90 )
# target_link_libraries( example_fortranInterface_daxpy PRIVATE tblas_c )

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
  echo "- example_cwrapper_gemm --------" &&
  ${CMAKE_CURRENT_BINARY_DIR}/cwrapper_gemm/example_cwrapper_gemm${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_fortranModule_caxpy --" &&
  ${CMAKE_CURRENT_BINARY_DIR}/fortranModule_caxpy/example_fortranModule_caxpy${CMAKE_EXECUTABLE_SUFFIX}
COMMAND
  echo "- example_fortranWrapper_saxpy -" &&
  ${CMAKE_CURRENT_BINARY_DIR}/fortranWrapper_ssymm/example_fortranWrapper_ssymm${CMAKE_EXECUTABLE_SUFFIX} )
