/// @file test_geqrf.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test GEQRF and UNMQR
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
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/unmqr.hpp>

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

    const T zero(0);

    idx_t m, n, nb;

    m = GENERATE(5, 10, 20);
    n = GENERATE(5, 10, 20);
    nb = GENERATE(1, 2, 4, 5);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100. * max(m, n)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, m, n);

    std::vector<T> tau(min(m, n));
    GeqrfOpts geqrfOpts;
    UnmqrOpts unmqrOpts;
    geqrfOpts.nb = nb;
    unmqrOpts.nb = nb;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(GENERAL, A, A_copy);
    real_t anorm = tlapack::lange(tlapack::MAX_NORM, A_copy);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " nb = " << nb)
    {
        geqrf(A, tau, geqrfOpts);

        // Copy upper triangular part of A to R
        laset(LOWER_TRIANGLE, zero, zero, R);
        lacpy(UPPER_TRIANGLE, slice(A, range(0, m), range(0, n)), R);

        // Test A == Q * R
        unmqr(LEFT_SIDE, NO_TRANS, A, tau, R, unmqrOpts);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A_copy(i, j) -= R(i, j);

        real_t repres = tlapack::lange(tlapack::MAX_NORM, A_copy);
        CHECK(repres <= tol * anorm);
    }
}
