/// @file test_trmm_blocked_mixed.cpp
/// @author Weslley S Pereira, National Renewable Energy Laboratory, USA
/// @brief Test TRMM blocked mixed
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// #define TLAPACK_USE_INTELAMX

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Main <T>LAPACK header
#include <tlapack/lapack/trmm_blocked_mixed.hpp>

// Auxiliary <T>LAPACK headers
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lantr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("TRMM blocked mixed works",
                   "[blas][trmm_blocked_mixed][trmm][blocked][mixed]",
                   (std::pair<float, float>),
                   (std::pair<double, float>),
                   (std::pair<float, Eigen::bfloat16>))
{
    using T1 = TestType::first_type;
    using T2 = TestType::second_type;

    using matrix1_t =
        tlapack::LegacyMatrix<T1, std::size_t, tlapack::Layout::ColMajor>;
    using matrix2_t =
        tlapack::LegacyMatrix<T2, std::size_t, tlapack::Layout::ColMajor>;

    using idx_t = size_type<matrix1_t>;
    typedef real_type<T1> real_t;

    // Functor
    Create<matrix1_t> new_matrix1;
    Create<matrix2_t> new_matrix2;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n = GENERATE(10000);
    idx_t k = GENERATE(1);
    idx_t nb = 32;

    DYNAMIC_SECTION("n = " << n)
    {
        std::vector<T2> A_;
        auto A = new_matrix2(A_, n, n);
        std::vector<T1> B_;
        auto B = new_matrix1(B_, n, k);
        std::vector<T1> E_;
        auto E = new_matrix1(E_, n, k);
        std::vector<T2> W_;
        auto W = new_matrix2(W_, nb, k);

        // Generate n-by-n upper-triangle random matrix
        mm.random(UPPER_TRIANGLE, A);

        // Generate n-by-k random matrix, and stabilize it
        mm.random(W);
        lacpy(GENERAL, W, B);

        // Compute the norm of A and X
        const real_t norma = lantr(ONE_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, A);
        const real_t normx = lange(ONE_NORM, B);

        // Copy B to E
        lacpy(GENERAL, B, E);

        // Solve A * X = B, storing the result in B
        trmm_blocked_mixed(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                           real_t(1), A, B, W, TrmmBlockedOpts(nb));

        // Compute the norm of B
        const real_t normb = lange(ONE_NORM, B);

        // Compute the error
        trmm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), A,
             E);
        for (idx_t j = 0; j < k; ++j)
            for (idx_t i = 0; i < n; ++i)
                E(i, j) -= B(i, j);
        const real_t error = lange(ONE_NORM, E);

        std::cout << "norma = " << norma << std::endl;
        std::cout << "normb = " << normb << std::endl;
        std::cout << "normx = " << normx << std::endl;
        std::cout << "error = " << error << std::endl;

        CHECK(error / ulp<real_t>() < n);
    }
}
