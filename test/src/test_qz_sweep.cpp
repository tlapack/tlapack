/// @file test_qz_sweep.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test qz sweep.
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
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
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/gghrd.hpp>
#include <tlapack/lapack/lahqz.hpp>
#include <tlapack/lapack/multishift_qz_sweep.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QZ sweep is backward stable",
                   "[generalized eigenvalues]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t n = GENERATE(12, 30);
    const idx_t ns = GENERATE(2, 4, 6);
    const idx_t ilo = GENERATE(0, 1);
    const idx_t ihioff = GENERATE(0, 1);
    const idx_t ihi = n - ihioff;
    const real_t zero(0);
    const real_t one(1);

    // Define the matrices
    std::vector<TA> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<TA> B_;
    auto B = new_matrix(B_, n, n);
    std::vector<TA> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);
    std::vector<TA> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<TA> Z_;
    auto Z = new_matrix(Z_, n, n);

    mm.hessenberg(A);
    mm.random(Uplo::Upper, B);

    // Make sure ilo and ihi correspond to the actual matrix
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = (TA)0.0;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i; ++j)
            A(i, j) = (TA)0.0;

    // Clear out subdiagonal
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            A(i, j) = zero;
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            B(i, j) = zero;

    lacpy(GENERAL, A, H);
    lacpy(GENERAL, B, T);
    std::vector<complex_t> alpha(ns);
    std::vector<TA> beta(ns);
    laset(GENERAL, zero, one, Q);
    laset(GENERAL, zero, one, Z);

    for (idx_t i = 0; i < ns; i++) {
        alpha[i] = ((complex_t)(TA)i);
        beta[i] = (TA)1.0;
    }

    DYNAMIC_SECTION(" n = " << n << " ns = " << ns << " ilo = " << ilo
                            << " ihi = " << ihi)
    {
        multishift_QZ_sweep(true, true, true, ilo, ihi, H, T, alpha, beta, Q,
                            Z);

        // Clean the lower triangular part that was used a workspace
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 4; i < n; ++i)
                H(i, j) = zero;
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 4; i < n; ++i)
                T(i, j) = zero;

        const real_t eps = uroundoff<real_t>();
        const real_t tol = real_t(n * 1.0e2) * eps;

        std::vector<TA> res_;
        auto res = new_matrix(res_, n, n);
        std::vector<TA> work_;
        auto work = new_matrix(work_, n, n);

        // Calculate residuals
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A);
        auto normA_res =
            check_generalized_similarity_transform(A, Q, Z, H, res, work);
        CHECK(normA_res <= tol * normA);

        auto normB = tlapack::lange(tlapack::FROB_NORM, B);
        auto normB_res =
            check_generalized_similarity_transform(B, Q, Z, T, res, work);
        CHECK(normB_res <= tol * normB);
    }
}
