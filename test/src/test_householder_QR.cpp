/// @file test_householder_QR.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/lapack/gen_householder_q.hpp>
#include <tlapack/lapack/householder_q_mul.hpp>
#include <tlapack/lapack/householder_qr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QR factorization of a general m-by-n matrix",
                   "[qr][qrf]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // Generate test case
    using variant_t = pair<HouseholderQRVariant, idx_t>;
    const variant_t variant =
        GENERATE((variant_t(HouseholderQRVariant::Blocked, 1)),
                 (variant_t(HouseholderQRVariant::Blocked, 2)),
                 (variant_t(HouseholderQRVariant::Blocked, 4)),
                 (variant_t(HouseholderQRVariant::Blocked, 5)),
                 (variant_t(HouseholderQRVariant::Level2, 1)));
    const idx_t m = GENERATE(5, 10, 20, 30);
    const idx_t n = GENERATE(5, 10, 20, 30);

    // Constants
    const idx_t nb = variant.second;
    const idx_t k = min(m, n);
    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(10. * max(m, n)) * eps;

    // Matrices and vectors
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, k);
    std::vector<T> R_;
    auto R = new_matrix(R_, m, n);
    std::vector<T> tau(k);

    // Generate random test case
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    // Copy A to A_copy
    lacpy(GENERAL, A, A_copy);
    // Compute norm of A
    real_t anorm = lange(MAX_NORM, A_copy);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " variant = "
                           << (char)variant.first << " nb = " << nb)
    {
        // QR decomposition
        HouseholderQROpts qrOpts;
        qrOpts.variant = variant.first;
        qrOpts.nb = nb;
        householder_qr(A, tau, qrOpts);

        // Copy A to Q and R
        lacpy(LOWER_TRIANGLE, slice(A, range(0, m), range(0, k)), Q);
        laset(LOWER_TRIANGLE, real_t(0), real_t(0), R);
        lacpy(UPPER_TRIANGLE, A, R);

        // Test Q is unitary
        gen_householder_q(FORWARD, COLUMNWISE_STORAGE, Q, tau,
                          GenHouseholderQOpts{(size_t)nb});
        auto orth_Q = check_orthogonality(Q);
        CHECK(orth_Q <= tol);

        // Test A == Q * R
        const auto V = slice(A, range(0, m), range(0, k));
        householder_q_mul(LEFT_SIDE, NO_TRANS, FORWARD, COLUMNWISE_STORAGE, V,
                          tau, R, HouseholderQMulOpts{(size_t)nb});
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A_copy(i, j) -= R(i, j);
        real_t repres = lange(MAX_NORM, A_copy);
        CHECK(repres <= tol * anorm);
    }
}
