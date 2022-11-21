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

#define TLAPACK_PREFERRED_MATRIX_LEGACY
#ifdef TLAPACK_TEST_EIGEN
    #include <tlapack/plugins/eigen.hpp>
#endif
#include <tlapack/plugins/legacyArray.hpp>

// 
// The matrix types that will be tested for routines
// that only accept real matrices
// 
#ifndef TLAPACK_REAL_TYPES_TO_TEST

    #define TLAPACK_LEGACY_REAL_TYPES_TO_TEST \
        (legacyMatrix<float,std::size_t,Layout::ColMajor>), \
        (legacyMatrix<double,std::size_t,Layout::ColMajor>), \
        (legacyMatrix<float,std::size_t,Layout::RowMajor>), \
        (legacyMatrix<double,std::size_t,Layout::RowMajor>)
    
    #ifdef TLAPACK_TEST_EIGEN
        #define TLAPACK_EIGEN_REAL_TYPES_TO_TEST \
            , \
            Eigen::MatrixXf, \
            Eigen::MatrixXd, \
            (Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>)
    #else
        #define TLAPACK_EIGEN_REAL_TYPES_TO_TEST
    #endif

    #define TLAPACK_REAL_TYPES_TO_TEST \
        TLAPACK_LEGACY_REAL_TYPES_TO_TEST \
        TLAPACK_EIGEN_REAL_TYPES_TO_TEST
#endif

// 
// The matrix types that will be tested for routines
// that only accept complex matrices
// 
#ifndef TLAPACK_COMPLEX_TYPES_TO_TEST

    #ifndef TLAPACK_LEGACY_COMPLEX_TYPES_TO_TEST
        #define TLAPACK_LEGACY_COMPLEX_TYPES_TO_TEST \
            (legacyMatrix<std::complex<float>,std::size_t,Layout::ColMajor>), \
            (legacyMatrix<std::complex<double>,std::size_t,Layout::ColMajor>), \
            (legacyMatrix<std::complex<float>,std::size_t,Layout::RowMajor>), \
            (legacyMatrix<std::complex<double>,std::size_t,Layout::RowMajor>)
    #endif
    
    #ifdef TLAPACK_TEST_EIGEN
        #define TLAPACK_EIGEN_COMPLEX_TYPES_TO_TEST \
            , \
            Eigen::MatrixXcf, \
            Eigen::MatrixXcd, \
            (Eigen::Matrix<std::complex<float>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>)
    #else
        #define TLAPACK_EIGEN_COMPLEX_TYPES_TO_TEST
    #endif

    #define TLAPACK_COMPLEX_TYPES_TO_TEST \
        TLAPACK_LEGACY_COMPLEX_TYPES_TO_TEST \
        TLAPACK_EIGEN_COMPLEX_TYPES_TO_TEST

#endif

// 
// List of matrix types that will be tested
// 
#ifndef TLAPACK_TYPES_TO_TEST
    #define TLAPACK_TYPES_TO_TEST \
        TLAPACK_REAL_TYPES_TO_TEST, TLAPACK_COMPLEX_TYPES_TO_TEST
#endif

#endif // TLAPACK_TESTDEFINITIONS_HH
