/// @file test_generalized_aed.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test AED for multishift QZ
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

// Other routines
#include <tlapack/lapack/aggressive_early_deflation_generalized.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("AED is backward stable",
                   "[generalized eigenvalues]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const T zero(0);
    const T one(1);
    const idx_t n = GENERATE(30, 100);
    const idx_t nw = GENERATE(2, 4, 6, 20);

    const real_t eps = uroundoff<real_t>();
    const real_t tol = real_t(1.0e2 * n) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> B_;
    auto B = new_matrix(B_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<T> Z_;
    auto Z = new_matrix(Z_, n, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, n, n);
    std::vector<T> B_copy_;
    auto B_copy = new_matrix(B_copy_, n, n);

    std::vector<complex_t> alpha(n);
    std::vector<T> beta(n);

    const idx_t ilo = 0;
    const idx_t ihi = n;

    // Generate random pencil in generalized Schur form
    mm.random(A);
    mm.random(B);

    // Zero out the lower triangular part
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            A(i, j) = zero;
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            B(i, j) = zero;

    lacpy(GENERAL, A, A_copy);
    lacpy(GENERAL, B, B_copy);
    laset(GENERAL, zero, one, Q);
    laset(GENERAL, zero, one, Z);

    DYNAMIC_SECTION("n = " << n << "nw = " << nw)
    {
        idx_t ns, nd;
        FrancisOpts opts;
        aggressive_early_deflation_generalized(true, true, true, ilo, ihi, nw,
                                               A, B, alpha, beta, Q, Z, ns, nd,
                                               opts);

        // Clean the lower triangular part that was used a workspace
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 4; i < n; ++i)
                A(i, j) = zero;
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 4; i < n; ++i)
                B(i, j) = zero;

        std::vector<T> res_;
        auto res = new_matrix(res_, n, n);
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);

        // Calculate residuals
        auto orth_res_norm_q = check_orthogonality(Q, res);
        CHECK(orth_res_norm_q <= tol);

        auto orth_res_norm_z = check_orthogonality(Z, res);
        CHECK(orth_res_norm_z <= tol);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A_copy);
        auto normA_res =
            check_generalized_similarity_transform(A_copy, Q, Z, A, res, work);
        CHECK(normA_res <= tol * normA);

        auto normB = tlapack::lange(tlapack::FROB_NORM, B_copy);
        auto normB_res =
            check_generalized_similarity_transform(B_copy, Q, Z, B, res, work);
        CHECK(normB_res <= tol * normB);
    }
}