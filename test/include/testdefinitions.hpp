/// @file testdefinitions.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// @brief Definitions for the unit tests
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
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
#ifdef TLAPACK_TEST_MDSPAN
    #include <tlapack/plugins/mdspan.hpp>
#endif
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>

#ifdef TLAPACK_TEST_MPFR
    #include <tlapack/plugins/mpreal.hpp>
#endif

//
// The matrix types that will be tested for routines
// that only accept real matrices
//
#ifndef TLAPACK_REAL_TYPES_TO_TEST

    #define TLAPACK_LEGACY_REAL_TYPES_TO_TEST                      \
        (legacyMatrix<float, std::size_t, Layout::ColMajor>),      \
            (legacyMatrix<double, std::size_t, Layout::ColMajor>), \
            (legacyMatrix<float, std::size_t, Layout::RowMajor>),  \
            (legacyMatrix<double, std::size_t, Layout::RowMajor>)

    #ifdef TLAPACK_TEST_MPFR
        #define TLAPACK_LEGACY_REAL_TYPES_TO_TEST_WITH_MPREAL \
            , legacyMatrix<mpfr::mpreal>
    #else
        #define TLAPACK_LEGACY_REAL_TYPES_TO_TEST_WITH_MPREAL
    #endif

    #ifdef TLAPACK_TEST_EIGEN
        #define TLAPACK_EIGEN_REAL_TYPES_TO_TEST                      \
            , Eigen::MatrixXf, Eigen::MatrixXd,                       \
                (Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, \
                               Eigen::RowMajor>),                     \
                (Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>)
    #else
        #define TLAPACK_EIGEN_REAL_TYPES_TO_TEST
    #endif

    #ifdef TLAPACK_TEST_MDSPAN
        #define TLAPACK_MDSPAN_REAL_TYPES_TO_TEST                       \
            ,                                                           \
                (std::experimental::mdspan<                             \
                    float, std::experimental::dextents<std::size_t, 2>, \
                    std::experimental::layout_left>),                   \
                (std::experimental::mdspan<                             \
                    float, std::experimental::dextents<std::size_t, 2>, \
                    std::experimental::layout_right>)
    #else
        #define TLAPACK_MDSPAN_REAL_TYPES_TO_TEST
    #endif

    #define TLAPACK_REAL_TYPES_TO_TEST                \
        TLAPACK_LEGACY_REAL_TYPES_TO_TEST             \
        TLAPACK_LEGACY_REAL_TYPES_TO_TEST_WITH_MPREAL \
        TLAPACK_EIGEN_REAL_TYPES_TO_TEST              \
        TLAPACK_MDSPAN_REAL_TYPES_TO_TEST
#endif

//
// The matrix types that will be tested for routines
// that only accept complex matrices
//
#ifndef TLAPACK_COMPLEX_TYPES_TO_TEST

    #ifndef TLAPACK_LEGACY_COMPLEX_TYPES_TO_TEST
        #define TLAPACK_LEGACY_COMPLEX_TYPES_TO_TEST             \
            (legacyMatrix<std::complex<float>, std::size_t,      \
                          Layout::ColMajor>),                    \
                (legacyMatrix<std::complex<double>, std::size_t, \
                              Layout::ColMajor>),                \
                (legacyMatrix<std::complex<float>, std::size_t,  \
                              Layout::RowMajor>),                \
                (legacyMatrix<std::complex<double>, std::size_t, \
                              Layout::RowMajor>)
    #endif

    #ifdef TLAPACK_TEST_EIGEN
        #define TLAPACK_EIGEN_COMPLEX_TYPES_TO_TEST                 \
            , Eigen::MatrixXcf, Eigen::MatrixXcd,                   \
                (Eigen::Matrix<std::complex<float>, Eigen::Dynamic, \
                               Eigen::Dynamic, Eigen::RowMajor>)
    #else
        #define TLAPACK_EIGEN_COMPLEX_TYPES_TO_TEST
    #endif

    #define TLAPACK_COMPLEX_TYPES_TO_TEST    \
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

#endif  // TLAPACK_TESTDEFINITIONS_HH
