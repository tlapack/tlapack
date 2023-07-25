/// @file test_geql2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test geql2
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
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/geql2.hpp>
#include <tlapack/lapack/ung2l.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QL factorization of a general m-by-n matrix",
                   "[ql]",
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

    const T zero(0);

    idx_t m, n, k;

    m = GENERATE(5, 6, 10, 20);
    n = GENERATE(5, 6, 10, 20);
    k = min(m, n);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100.0 * max(m, n)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, k);

    std::vector<T> tau(k);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(GENERAL, A, A_copy);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        // QLs factorization
        geql2(A, tau);

        // Generate Q
        lacpy(GENERAL, slice(A, range(0, m), range(n - k, n)), Q);
        ung2l(Q, tau);

        // Check orthogonality of Q
        std::vector<T> Wq_;
        auto Wq = new_matrix(Wq_, k, k);
        auto orth_Q = check_orthogonality(Q, Wq);
        CHECK(orth_Q <= tol);

        // Set upper triangular part of A to zero to get L
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = j + 1; i < m; ++i)
                A(m - i - 1, n - j - 1) = zero;
        }
        auto L = slice(A, range(m - k, m), range(0, n));

        // Test A_copy = Q * L
        std::vector<T> A2_;
        auto A2 = new_matrix(A2_, k, n);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1.), Q, A_copy, A2);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < k; ++i)
                A2(i, j) -= L(i, j);

        real_t repres = lange(MAX_NORM, A2);
        CHECK(repres <= tol);
    }
}
