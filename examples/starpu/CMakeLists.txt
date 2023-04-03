# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( starpu CXX )
set(CMAKE_CXX_STANDARD 20)

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# Load StarPU
find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.4)
if (STARPU_FOUND)
  include_directories (${STARPU_INCLUDE_DIRS})
  link_directories (${STARPU_STATIC_LIBRARY_DIRS})
  link_libraries (${STARPU_STATIC_LIBRARIES})
else (STARPU_FOUND)
  message(FATAL_ERROR "StarPU not found")
endif()

# add the example eigen
add_executable( example_starpu example_starpu.cpp )
target_link_libraries( example_starpu PRIVATE tlapack )
