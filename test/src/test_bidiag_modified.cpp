/// @file test_bidiag.cpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test bidiagonal reduction
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/bidiag.hpp>
#include <tlapack/lapack/ungbr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("bidiagonal reduction is backward stable",
                   "[bidiagonal][svd]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    using variant_t = pair<BidiagVariant, idx_t>;
    const variant_t variant = GENERATE((variant_t(BidiagVariant::Blocked, 1)),
                                       (variant_t(BidiagVariant::Blocked, 2)),
                                       (variant_t(BidiagVariant::Blocked, 5)),
                                       (variant_t(BidiagVariant::Level2, 1)));
    const idx_t m = GENERATE(1, 4, 5, 10, 15);
    const idx_t n = GENERATE(1, 4, 5, 10, 12);
    const idx_t k = min(m, n);
    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(10. * max(m, n)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, k);
    std::vector<T> Z_;
    auto Z = new_matrix(Z_, k, n);

    std::vector<T> tauv(k);
    std::vector<T> tauw(k);

    // Generate random m-by-n matrix
    mm.random(A);

    lacpy(GENERAL, A, A_copy);
    real_t normA = lange(MAX_NORM, A);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " variant = "
                           << (char)variant.first << " nb = " << variant.second)
    {
        BidiagOpts bidiagOpts;
        bidiagOpts.variant = variant.first;
        bidiagOpts.nb = variant.second;
        bidiag(A, tauv, tauw, bidiagOpts);

        // Get bidiagonal B
        std::vector<T> B_;
        auto B = new_matrix(B_, k, k);
        laset(GENERAL, real_t(0), real_t(0), B);

        if (m >= n) {
            // copy upper bidiagonal matrix
            B(0, 0) = A(0, 0);
            for (idx_t j = 1; j < k; ++j) {
                B(j - 1, j) = A(j - 1, j);
                B(j, j) = A(j, j);
            }
        }
        else {
            // copy lower bidiagonal matrix
            B(0, 0) = A(0, 0);
            for (idx_t j = 1; j < k; ++j) {
                B(j, j - 1) = A(j, j - 1);
                B(j, j) = A(j, j);
            }
        }

        // Generate m-by-k unitary matrix Q
        UngbrOpts ungbrOpts;
        ungbrOpts.nb = variant.second;
        lacpy(LOWER_TRIANGLE, slice(A, range{0, m}, range{0, k}), Q);
        ungbr_q(n, Q, tauv, ungbrOpts);

        // Test for Q's orthogonality
        std::vector<T> Wq_;
        auto Wq = new_matrix(Wq_, k, k);
        auto orth_Q = check_orthogonality(Q, Wq);
        CHECK(orth_Q <= tol);

        lacpy(UPPER_TRIANGLE, slice(A, range{0, k}, range{0, n}), Z);
        ungbr_p(m, Z, tauw, ungbrOpts);

        // Test for Z's orthogonality
        std::vector<T> Wz_;
        auto Wz = new_matrix(Wz_, k, k);
        auto orth_Z = check_orthogonality(Z, Wz);
        CHECK(orth_Z <= tol);

        // Test Q * B * Z^H = A
        std::vector<T> K_;
        auto K = new_matrix(K_, m, k);
        gemm(NO_TRANS, NO_TRANS, real_t(1.), Q, B, K);
        gemm(NO_TRANS, NO_TRANS, real_t(1.), K, Z, real_t(-1.), A_copy);
        real_t repres = lange(MAX_NORM, A_copy);
        CHECK(repres <= tol * normA);
    }
}
