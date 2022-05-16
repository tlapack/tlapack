# Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

configure_file(version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/version.h)
add_executable(version version.cpp)
target_include_directories(version PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
