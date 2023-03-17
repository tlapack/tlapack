/// @file test_latrs.cpp Test safe scaling linear solve
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

// Other routines
#include <tlapack/blas/trmv.hpp>
#include <tlapack/lapack/latrs.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("safe scaling solve", "[latrs]", TLAPACK_TYPES_TO_TEST)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    Create<matrix_t> new_matrix;

    const T zero(0);
    const T one(1);

    // Number of rows in the matrix
    idx_t n = GENERATE(10, 20, 30);

    const real_t eps = uroundoff<real_t>();
    const real_t tol = real_t(n * 1.0e2) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> x(n);
    std::vector<T> b(n);
    real_t scale;
    std::vector<real_t> cnorm(n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < j + 1; ++i)
            A(i, j) = rand_helper<T>();
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = zero;

    for (idx_t i = 0; i < n; ++i)
        b[i] = rand_helper<T>();

    for (idx_t i = 0; i < n; ++i)
        x[i] = b[i];

    latrs(Uplo::Upper, Op::NoTrans, Diag::NonUnit, false, A, x, scale, cnorm);

    trmv(Uplo::Upper, Op::NoTrans, Diag::NonUnit, A, x);

    real_t bnorm = tlapack::lange(Norm::Max, b);
    real_t enorm = real_t(0);
    for (idx_t i = 0; i < n; ++i)
        enorm += abs1(x[i] - scale * b[i]);

    CHECK(enorm <= tol * bnorm);
}
