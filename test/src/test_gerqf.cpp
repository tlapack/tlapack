/// @file test_gerqf.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test gerqf
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
#include <tlapack/lapack/gerqf.hpp>
#include <tlapack/lapack/ungr2.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("RQ factorization of a general m-by-n matrix",
                   "[rq]",
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

    idx_t m, n, k, nb;

    m = GENERATE(5, 10, 20);
    n = GENERATE(5, 10, 20);
    nb = GENERATE(1, 2, 3, 4, 5);
    k = min(m, n);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100.0 * max(m, n)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, k, n);

    std::vector<T> tau(min(m, n));

    // Workspace computation:
    gerqf_opts_t<idx_t> gerqfOpts;
    gerqfOpts.nb = nb;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        // RQ factorization
        gerqf(A, tau, gerqfOpts);

        // Generate Q
        lacpy(Uplo::General, slice(A, range(m - k, m), range(0, n)), Q);
        ungr2(Q, tau);

        // Check orthogonality of Q
        std::vector<T> Wq_;
        auto Wq = new_matrix(Wq_, k, k);
        auto orth_Q = check_orthogonality(Q, Wq);
        CHECK(orth_Q <= tol);

        // Set lower triangular part of A to zero to get R
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i + 1 < min(k, n - j); ++i)
                A(m - 1 - i, j) = zero;
        auto R = slice(A, range(0, m), range(n - k, n));

        // Test A_copy = R * Q
        std::vector<T> A2_;
        auto A2 = new_matrix(A2_, m, k);
        gemm(Op::NoTrans, Op::ConjTrans, real_t(1.), A_copy, Q, A2);
        for (idx_t j = 0; j < k; ++j)
            for (idx_t i = 0; i < m; ++i)
                A2(i, j) -= R(i, j);

        real_t repres = lange(Norm::Max, A2);
        CHECK(repres <= tol);
    }
}
