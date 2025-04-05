/// @file test_generalized_schur_swap.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test 1x1 and 2x2 generalized schur swaps
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
#include <tlapack/lapack/generalized_schur_swap.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("generalized schur swap gives correct result",
                   "[generalized eigenvalues]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const T zero(0);
    const T one(1);
    idx_t n = 10;

    const idx_t j = GENERATE(0, 1, 6);
    const idx_t n1 = GENERATE(1, 2);
    const idx_t n2 = GENERATE(1, 2);

    if (is_real<T> || (n1 == 1 && n2 == 1)) {
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

        // Generate random matrix in Schur form
        mm.random(A);
        mm.random(B);

        // Set lower triangular part to zero
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < n; ++i)
                A(i, j) = zero;
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < n; ++i)
                B(i, j) = zero;

        if (n1 == 2) A(j + 1, j) = rand_helper<T>(mm.gen);
        if (n2 == 2) A(j + n1 + 1, j + n1) = rand_helper<T>(mm.gen);

        lacpy(GENERAL, A, A_copy);
        lacpy(GENERAL, B, B_copy);
        laset(GENERAL, zero, one, Q);
        laset(GENERAL, zero, one, Z);

        DYNAMIC_SECTION("j = " << j << " n1 = " << n1 << " n2 =" << n2)
        {
            int ierr =
                generalized_schur_swap(true, true, A, B, Q, Z, j, n1, n2);
            CHECK(ierr == 0);
            // Calculate residuals

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
            auto normA_res = check_generalized_similarity_transform(
                A_copy, Q, Z, A, res, work);
            CHECK(normA_res <= tol * normA);

            auto normB = tlapack::lange(tlapack::FROB_NORM, B_copy);
            auto normB_res = check_generalized_similarity_transform(
                B_copy, Q, Z, B, res, work);
            CHECK(normB_res <= tol * normB);
        }
    }
}