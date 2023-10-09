/// @file test_svd_qr.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test implicit QR variation of SVD
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// tlapack routines
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/svd_qr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("svd is backward stable",
                   "[qr-svd][svd]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const real_t zero(0);
    const real_t one(1);

    idx_t n;

    n = GENERATE(1, 2, 4, 5, 10, 12, 20);

    const real_t eps = ulp<real_t>();
    real_t tol = real_t(20. * n) * eps;
    // Use a slightly larger tolerance for half precision
    if (eps > real_t(1.0e-6)) tol = tol * real_t(5.);

    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<T> Pt_;
    auto Pt = new_matrix(Pt_, n, n);

    std::vector<real_t> d(n);
    std::vector<real_t> e(n - 1);
    std::vector<real_t> d_copy(n);
    std::vector<real_t> e_copy(n - 1);

    // Generate random bidiagonal matrix
    for (idx_t j = 0; j < n; ++j)
        d[j] = rand_helper<real_t>();
    for (idx_t j = 0; j + 1 < n; ++j)
        e[j] = rand_helper<real_t>();

    copy(d, d_copy);
    copy(e, e_copy);

    laset(Uplo::General, zero, one, Q);
    laset(Uplo::General, zero, one, Pt);

    DYNAMIC_SECTION(" n = " << n)
    {
        int err = svd_qr(Uplo::Upper, true, true, d, e, Q, Pt);
        REQUIRE(err == 0);

        // Check that singular values are positive and sorted in decreasing
        // order
        for (idx_t i = 0; i < n; ++i) {
            CHECK(d[i] >= zero);
        }
        for (idx_t i = 0; i < n - 1; ++i) {
            CHECK(d[i] >= d[i + 1]);
        }

        // Test for Q's orthogonality
        std::vector<T> Wq_;
        auto Wq = new_matrix(Wq_, n, n);
        auto orth_Q = check_orthogonality(Q, Wq);
        CHECK(orth_Q <= tol);

        // Test for Pt's orthogonality
        std::vector<T> Wpt_;
        auto Wpt = new_matrix(Wpt_, n, n);
        auto orth_Pt = check_orthogonality(Pt, Wpt);
        CHECK(orth_Pt <= tol);

        // Test Q * B * Z^H = A
        std::vector<T> B_;
        auto B = new_matrix(B_, n, n);
        laset(Uplo::General, zero, zero, B);
        for (idx_t j = 0; j < n; ++j) {
            B(j, j) = d[j];
        }
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        laset(Uplo::General, zero, zero, A);
        A(0, 0) = d_copy[0];
        for (idx_t j = 1; j < n; ++j) {
            A(j - 1, j) = e_copy[j - 1];
            A(j, j) = d_copy[j];
        }
        real_t normA = tlapack::lange(tlapack::Norm::Max, A);
        std::vector<T> K_;
        auto K = new_matrix(K_, n, n);
        laset(Uplo::General, zero, zero, K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), B, Pt, real_t(0), K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), Q, K, real_t(-1.), A);
        real_t repres = lange(Norm::Max, A);
        CHECK(repres <= tol * normA);
    }
}
