# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file

include( "${TLAPACK_SOURCE_DIR}/cmake/FetchPackage.cmake" )

#-------------------------------------------------------------------------------
# Load testBLAS
FetchPackage( "testBLAS" "https://github.com/tlapack/testBLAS.git" "master" )

#-------------------------------------------------------------------------------
# Build BLAS++ tests
if( BUILD_BLASPP_TESTS )
  add_subdirectory( blaspp )
endif()

#-------------------------------------------------------------------------------
# Build LAPACK++ tests
if( BUILD_LAPACKPP_TESTS )
  add_subdirectory( lapackpp )
endif()