/// @file test_trmm_blocked_mixed.cpp
/// @author Weslley S Pereira, National Renewable Energy Laboratory, USA
/// @brief Test TRMM blocked mixed
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Main <T>LAPACK header
#include <tlapack/lapack/trmm_blocked_mixed.hpp>

// Auxiliary <T>LAPACK headers
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lantr.hpp>

#if __has_include(<stdfloat>) && __cplusplus > 202002L
    #define TEST_TYPES_bTRMM                     \
        (std::tuple<float, float, double>),      \
            (std::tuple<double, float, double>), \
            (std::tuple<float, std::bfloat16_t, double>)
#else
    #define TEST_TYPES_bTRMM \
        (std::tuple<float, float, double>), (std::tuple<double, float, double>)
#endif

using namespace tlapack;

TEMPLATE_TEST_CASE("TRMM blocked mixed works",
                   "[blas][trmm_blocked_mixed][trmm][blocked][mixed]",
                   TEST_TYPES_bTRMM)
{
    using T1 = std::tuple_element<0, TestType>::type;
    using T2 = std::tuple_element<1, TestType>::type;
    using Tref = std::tuple_element<2, TestType>::type;

    using matrix1_t =
        tlapack::LegacyMatrix<T1, std::size_t, tlapack::Layout::ColMajor>;
    using matrix2_t =
        tlapack::LegacyMatrix<T2, std::size_t, tlapack::Layout::ColMajor>;
    using matrixRef_t =
        tlapack::LegacyMatrix<Tref, std::size_t, tlapack::Layout::ColMajor>;

    using idx_t = size_type<matrix1_t>;
    typedef real_type<T1> real_t;
    typedef real_type<T1> realRef_t;

    // Functor
    Create<matrix1_t> new_matrix1;
    Create<matrix2_t> new_matrix2;
    Create<matrixRef_t> new_matrixRef;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n = GENERATE(64, 128, 256);
    idx_t m = GENERATE(64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                       65536);
    idx_t nb = 128;

    DYNAMIC_SECTION("m = " << m << " n = " << n << " nb = " << nb)
    {
        std::cout << "m = " << m << " n = " << n << " nb = " << nb << std::endl;

        std::vector<T2> A_;
        auto A = new_matrix2(A_, m, m);
        std::vector<T2> Xlow_;
        auto Xlow = new_matrix2(Xlow_, m, n);
        std::vector<T2> W_;
        auto W = new_matrix2(W_, nb, n);

        std::vector<T1> B_;
        auto B = new_matrix1(B_, m, n);
        std::vector<T1> Ahigh_;
        auto Ahigh = new_matrix1(Ahigh_, m, m);
        std::vector<T1> Bhigh_;
        auto Bhigh = new_matrix1(Bhigh_, m, n);

        std::vector<Tref> Aref_;
        auto Aref = new_matrixRef(Aref_, m, m);
        std::vector<Tref> Bref_;
        auto Bref = new_matrixRef(Bref_, m, n);
        std::vector<Tref> E_;
        auto E = new_matrixRef(E_, m, n);
        std::vector<Tref> E2_;
        auto E2 = new_matrixRef(E2_, m, m);

        // Generate m-by-m upper-triangle random matrix
        mm.randn(UPPER_TRIANGLE, Aref);

        // Generate m-by-n random matrix in the reference precision
        mm.randn(Bref);

        // Copy matrices
        lacpy(GENERAL, Aref, A);
        lacpy(GENERAL, Aref, Ahigh);
        lacpy(GENERAL, Bref, Bhigh);
        lacpy(GENERAL, Bref, B);
        lacpy(GENERAL, Bref, Xlow);

        // Compute the approximation errors in the input

        const realRef_t normx = lange(ONE_NORM, Bref);
        const realRef_t norma =
            lantr(ONE_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, Aref);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                E(i, j) = Bref(i, j) - B(i, j);
        const realRef_t errXhigh = lange(ONE_NORM, E) / normx;

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                E(i, j) = Bref(i, j) - Xlow(i, j);
        const realRef_t errXlow = lange(ONE_NORM, E) / normx;

        for (idx_t j = 0; j < m; ++j)
            for (idx_t i = 0; i <= j; ++i)
                E2(i, j) = Aref(i, j) - Ahigh(i, j);
        const realRef_t errAhigh =
            lantr(ONE_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, E2) / norma;

        for (idx_t j = 0; j < m; ++j)
            for (idx_t i = 0; i <= j; ++i)
                E2(i, j) = Aref(i, j) - A(i, j);
        const realRef_t errAlow =
            lantr(ONE_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, E2) / norma;

        std::cout << "Relative error on A when cast to T1 = "
                  << errAhigh / norma << std::endl;
        std::cout << "Relative error on A when cast to T2 = " << errAlow / norma
                  << std::endl;
        std::cout << "Relative error on X when cast to T1 = "
                  << errXhigh / normx << std::endl;
        std::cout << "Relative error on X when cast to T2 = " << errXlow / normx
                  << std::endl;

        // Solve A * X = B in mixed precision, storing the result in B
        trmm_blocked_mixed(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                           real_t(1), A, B, W, TrmmBlockedOpts(nb));

        // Solve A * X = B in high precision, storing the result in Bhigh
        trmm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1),
             Ahigh, Bhigh);

        // Solve A * X = B in the reference precision, storing the result in
        // Bref
        trmm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1),
             Aref, Bref);

        // Compute the errors
        lacpy(GENERAL, Bref, E);
        const real_t normb = lange(ONE_NORM, Bref);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i) {
                Bref(i, j) -= Bhigh(i, j);
                E(i, j) -= B(i, j);
            }
        const real_t normEhigh = lange(ONE_NORM, Bref);
        const real_t normEmixed = lange(ONE_NORM, E);

        std::cout << "Relative error using the single precision algorithm = "
                  << normEhigh / normb << std::endl;
        std::cout << "Relative error using the mixed precision algorithm = "
                  << normEmixed / normb << std::endl;
    }
}
