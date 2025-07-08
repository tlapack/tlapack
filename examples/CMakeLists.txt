# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Add the CMake directory for custom CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

add_custom_target( run-all-examples DEPENDS run-all-examples-cmd )
add_custom_command(
  OUTPUT run-all-examples-cmd
  COMMAND echo "Running all examples:" && echo "" )

# add the example geqr2
add_subdirectory( geqr2 )
add_custom_command(
  OUTPUT run-all-examples-cmd APPEND
  COMMAND
    echo "- example_geqr2 ----------------" &&
    "${CMAKE_CURRENT_BINARY_DIR}/geqr2/example_geqr2${CMAKE_EXECUTABLE_SUFFIX}"
    && echo ""
)
# add the example geqrtr
add_subdirectory( geqrtr )
add_custom_command(
  OUTPUT run-all-examples-cmd APPEND
  COMMAND
    echo "- example_geqrtr ----------------" &&
    "${CMAKE_CURRENT_BINARY_DIR}/geqrtr/example_geqrtr${CMAKE_EXECUTABLE_SUFFIX}"
    && echo ""
)

# add the example laed4
# add_subdirectory( laed4 )
# add_custom_command(
#   OUTPUT run-all-examples-cmd APPEND
#   COMMAND
#     echo "- example_laed4 ----------------" &&
#     "${CMAKE_CURRENT_BINARY_DIR}/laed4/example_laed4${CMAKE_EXECUTABLE_SUFFIX}"
#     && echo ""
# )

# add the eigenvalues example
add_subdirectory( eigenvalues )
add_custom_command(
  OUTPUT run-all-examples-cmd APPEND
  COMMAND
    echo "- example_eigenvalues ----------------" &&
    "${CMAKE_CURRENT_BINARY_DIR}/eigenvalues/example_eigenvalues${CMAKE_EXECUTABLE_SUFFIX}"
    && echo ""
)

# add the debug example
add_subdirectory( cpp_visualizer )

# add the example gemm (Use MPFR library if it is available)
add_subdirectory( gemm )
add_custom_command(
  OUTPUT run-all-examples-cmd APPEND
  COMMAND
    echo "- example_gemm ----------------" &&
    "${CMAKE_CURRENT_BINARY_DIR}/gemm/example_gemm${CMAKE_EXECUTABLE_SUFFIX}"
    && echo ""
)

# add the example lu (Use MPFR library if it is available)
add_subdirectory( lu )
add_custom_command(
  OUTPUT run-all-examples-cmd APPEND
  COMMAND
    echo "- example_lu ----------------" &&
    "${CMAKE_CURRENT_BINARY_DIR}/lu/example_lu${CMAKE_EXECUTABLE_SUFFIX}"
    && echo ""
)

# add the example access_types
add_subdirectory( access_types )
add_custom_command(
  OUTPUT run-all-examples-cmd APPEND
  COMMAND
    echo "- example_accessTypes ----------------" &&
    "${CMAKE_CURRENT_BINARY_DIR}/access_types/example_accessTypes${CMAKE_EXECUTABLE_SUFFIX}"
    && echo ""
)

# add the example potrf
find_package( LAPACK QUIET )
if( LAPACK_FOUND )
  find_package( LAPACKE QUIET )
  if( LAPACKE_FOUND )
    add_subdirectory( potrf )
    add_custom_command(
      OUTPUT run-all-examples-cmd APPEND
      COMMAND
        echo "- example_potrf ----------------" &&
        "${CMAKE_CURRENT_BINARY_DIR}/potrf/example_potrf${CMAKE_EXECUTABLE_SUFFIX}"
        && echo ""
    )
  else()
    mark_as_advanced( FORCE LAPACKE_DIR )
  endif()
else()
  mark_as_advanced( FORCE LAPACK_DIR )
endif()

find_package( Eigen3 QUIET )
if( Eigen3_FOUND )
  # add the example eigen
  add_subdirectory( eigen )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- example_eigen ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/eigen/example_eigen${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
  # add the example performance_eigen
  add_subdirectory( performance_eigen )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- performance_eigen ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/performance_eigen/performance_eigen${CMAKE_EXECUTABLE_SUFFIX}"
      && echo "-x-x-x-x-x-x-x-x-" &&
      "${CMAKE_CURRENT_BINARY_DIR}/performance_eigen/performance_tlapack${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
  # add the example create_float_library
  add_subdirectory( create_float_library )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- example_create_float_library ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/create_float_library/example_create_float_library${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
else()
  mark_as_advanced( FORCE Eigen3_DIR )
endif()

find_package( mdspan QUIET )
if( mdspan_FOUND )
  # add the example mdspan
  add_subdirectory( mdspan )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- example_mdspan ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/mdspan/example_mdspan${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
else()
  mark_as_advanced( FORCE mdspan_DIR )
endif()

# add the example gemm using the C wrapper
if( BUILD_C_WRAPPERS )
  add_subdirectory( cwrapper_gemm )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- cwrapper_gemm ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/cwrapper_gemm/example_cwrapper_gemm${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
endif()

# add the example caxpy using the Fortran module
if( BUILD_Fortran_WRAPPERS )
  add_subdirectory( fortranModule_caxpy )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- fortranModule_caxpy ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/fortranModule_caxpy/example_fortranModule_caxpy${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
endif()

# add the example daxpy using the Fortran wrapper
if( BUILD_Fortran_WRAPPERS )
  add_subdirectory( fortranModule_ssymm )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- fortranModule_ssymm ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/fortranModule_ssymm/example_fortranModule_ssymm${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
endif()

# add the example starpu
find_package( PkgConfig QUIET )
if (PkgConfig_FOUND)
  pkg_check_modules(STARPU QUIET starpu-1.4)
  if (STARPU_FOUND)
    add_subdirectory( starpu )
    add_custom_command(
      OUTPUT run-all-examples-cmd APPEND
      COMMAND
        echo "- example_starpu_lu ----------------" &&
        "${CMAKE_CURRENT_BINARY_DIR}/starpu/example_starpu_lu${CMAKE_EXECUTABLE_SUFFIX}"
        && echo ""
    )
  endif()
endif()

# add the example Extreme interoperability
find_package( mdspan QUIET )
find_package( Eigen3 QUIET )
if( mdspan_FOUND AND Eigen3_FOUND )
  add_subdirectory( extreme_interoperability )
  add_custom_command(
    OUTPUT run-all-examples-cmd APPEND
    COMMAND
      echo "- example_interoperability ----------------" &&
      "${CMAKE_CURRENT_BINARY_DIR}/extreme_interoperability/example_interoperability${CMAKE_EXECUTABLE_SUFFIX}"
      && echo ""
  )
else()
  mark_as_advanced( FORCE mdspan_DIR )
  mark_as_advanced( FORCE Eigen3_DIR )
endif()
