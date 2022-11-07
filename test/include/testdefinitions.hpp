/// @file testdefinitions.hpp
/// @brief Definitions for the unit tests
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TESTDEFINITIONS_HH
#define TLAPACK_TESTDEFINITIONS_HH

#include <complex>
#include <tlapack/plugins/legacyArray.hpp>

// 
// The matrix types that will be tested for routines
// that only accept real matrices
// 
#ifndef TLAPACK_REAL_TYPES_TO_TEST
    #define TLAPACK_REAL_TYPES_TO_TEST \
        (legacyMatrix<float,std::size_t,Layout::ColMajor>), \
        (legacyMatrix<double,std::size_t,Layout::ColMajor>), \
        (legacyMatrix<float,std::size_t,Layout::RowMajor>), \
        (legacyMatrix<double,std::size_t,Layout::RowMajor>)
#endif

// 
// The matrix types that will be tested for routines
// that only accept complex matrices
// 
#ifndef TLAPACK_COMPLEX_TYPES_TO_TEST
    #define TLAPACK_COMPLEX_TYPES_TO_TEST \
        (legacyMatrix<std::complex<float>,std::size_t,Layout::ColMajor>), \
        (legacyMatrix<std::complex<double>,std::size_t,Layout::ColMajor>), \
        (legacyMatrix<std::complex<float>,std::size_t,Layout::RowMajor>), \
        (legacyMatrix<std::complex<double>,std::size_t,Layout::RowMajor>)
#endif

// 
// List of matrix types that will be tested
// 
#ifndef TLAPACK_TYPES_TO_TEST
    #define TLAPACK_TYPES_TO_TEST \
        TLAPACK_REAL_TYPES_TO_TEST, TLAPACK_COMPLEX_TYPES_TO_TEST
#endif

#endif // TLAPACK_TESTDEFINITIONS_HH
