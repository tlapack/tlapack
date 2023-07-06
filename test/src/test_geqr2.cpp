/// @file test_geqrf.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test GEQR2 and UNG2R
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
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/ung2r.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QR factorization of a general m-by-n matrix",
                   "[qr][qr2]",
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

    m = GENERATE(10, 20, 30);
    n = GENERATE(10, 20, 30);
    k = min(m, n);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(m * n) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, k);

    std::vector<T> tau(min(m, n));

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        geqr2(A, tau);

        // Q is sliced down to the desired size of output Q (m-by-k).
        // It stores the desired number of Householder reflectors that UNG2R
        // will use.
        lacpy(Uplo::General, slice(A, range(0, m), range(0, k)), Q);

        ung2r(Q, tau);

        std::vector<T> orthres_;
        auto orthres = new_matrix(orthres_, k, k);
        auto orth_Q = check_orthogonality(Q, orthres);
        CHECK(orth_Q <= tol);

        // R is sliced from A after
        std::vector<T> R_;
        auto R = new_matrix(R_, k, n);
        laset(Uplo::Lower, zero, zero, R);
        lacpy(Uplo::Upper, slice(A, range(0, k), range(0, n)), R);

        // Test A = Q * R
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), Q, R, real_t(-1.), A_copy);

        real_t repres = tlapack::lange(tlapack::Norm::Max, A_copy);
        CHECK(repres <= tol);
    }
}
