/// @file test_trmm_blocked_mixed.cpp
/// @author Weslley S Pereira, National Renewable Energy Laboratory, USA
/// @brief Test TRMM blocked mixed
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
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
    #define TEST_TYPES_bTRMM                 \
        (std::tuple<double, float, double>), \
            (std::tuple<float, std::bfloat16_t, double>)
#else
    #define TEST_TYPES_bTRMM (std::tuple<double, float, double>)
#endif

using namespace tlapack;

TEMPLATE_TEST_CASE("TRMM blocked mixed works",
                   "[blas][trmm_blocked_mixed][trmm][blocked][mixed]",
                   TEST_TYPES_bTRMM)
{
    using T = typename std::tuple_element<0, TestType>::type;
    using Tlow = typename std::tuple_element<1, TestType>::type;
    using Tref = typename std::tuple_element<2, TestType>::type;

    using matrix_t =
        tlapack::LegacyMatrix<T, std::size_t, tlapack::Layout::ColMajor>;
    using matrixLow_t =
        tlapack::LegacyMatrix<Tlow, std::size_t, tlapack::Layout::ColMajor>;
    using matrixRef_t =
        tlapack::LegacyMatrix<Tref, std::size_t, tlapack::Layout::ColMajor>;

    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;
    typedef real_type<Tref> realRef_t;

    // Functor
    Create<matrix_t> new_matrix;
    Create<matrixLow_t> new_matrixLow;
    Create<matrixRef_t> new_matrixRef;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n = GENERATE(64, 128, 256);
    idx_t m = GENERATE(64, 128, 256, 512, 1024, 2048);
    idx_t nb = 128;

    const real_t u = uroundoff<real_t>();
    const real_t delta2m = (2 * real_t(m) * u) / (1 - 2 * real_t(m) * u);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " nb = " << nb)
    {
        std::vector<Tlow> Alow_;
        auto Alow = new_matrixLow(Alow_, m, m);
        std::vector<Tlow> W_;
        auto W = new_matrixLow(W_, nb, n);

        std::vector<T> A_;
        auto A = new_matrix(A_, m, m);
        std::vector<T> B_;
        auto B = new_matrix(B_, m, n);
        std::vector<T> C_;
        auto C = new_matrix(C_, m, n);
        auto& X = B;

        std::vector<Tref> Aref_;
        auto Aref = new_matrixRef(Aref_, m, m);
        std::vector<Tref> Bref_;
        auto Bref = new_matrixRef(Bref_, m, n);

        std::vector<Tref> E0_;
        auto E0 = new_matrixRef(E0_, m, m);
        std::vector<Tref> E1_;
        auto E1 = new_matrixRef(E1_, m, n);
        auto& E2 = Bref;

        // Generate m-by-m upper-triangle random matrix
        mm.randn(UPPER_TRIANGLE, A);

        // Generate m-by-n random matrix
        mm.randn(X);

        // Copy matrices
        lacpy(GENERAL, A, Alow);
        lacpy(GENERAL, A, Aref);
        lacpy(GENERAL, X, C);
        lacpy(GENERAL, X, Bref);

        // Compute the approximation errors in the input matrix A

        const realRef_t norma =
            lantr(ONE_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, Aref);

        for (idx_t j = 0; j < m; ++j)
            for (idx_t i = 0; i <= j; ++i)
                E0(i, j) = Aref(i, j) - Alow(i, j);
        const realRef_t errA =
            lantr(ONE_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, E0) / norma;

        INFO("Relative error on A when cast to Tlow = " << errA);

        // Compute Alow * X in mixed precision, storing the result in C
        trmm_blocked_mixed(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                           real_t(1), Alow, C, W, TrmmBlockedOpts{nb});

        // Compute A * X in precision T, storing the result in B
        trmm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), A,
             B);

        // Compute A * X in reference precision, storing the result in Bref
        trmm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, realRef_t(1),
             Aref, Bref);

        // Prepare for the error computation
        // E2 is a reference to Bref
        lacpy(GENERAL, Bref, E1);
        const real_t normb = lange(ONE_NORM, Bref);

        // Compute the errors
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i) {
                E1(i, j) -= B(i, j);
                E2(i, j) -= C(i, j);
            }
        const real_t normE1 = lange(ONE_NORM, E1) / normb;
        const real_t normE2 = lange(ONE_NORM, E2) / normb;

        INFO("Relative error on B = " << normE1);
        INFO("Relative error on C (uses mixed precision) = " << normE2);

        CHECK(normE1 <= delta2m);
    }
}
