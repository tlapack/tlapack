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
#include <tlapack/plugins/debugutils.hpp>

using namespace tlapack;

template <class matrix_t, class vector_t>
void generate_latrs_test_matrix(
    const int imat, Uplo uplo, matrix_t& A, vector_t& b, Diag& diag)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;

    const T zero(0);
    const real_t one(1);

    real_t unfl = safe_min<real_t>();
    real_t eps = ulp<real_t>();
    real_t smlnum = unfl;
    real_t bignum = one / smlnum;

    real_t badc2 = real_t(0.1) / eps;
    real_t badc1 = sqrt(badc2);

    idx_t n = ncols(A);

    if ((imat >= 7 and imat <= 10) or imat == 18) {
        diag = Diag::Unit;
    }
    else {
        diag = Diag::NonUnit;
    }

    if (n <= 0) return;

    idx_t kl, ku;
    real_t cndnum, anorm;

    if (imat == 1 or imat == 7) {
        kl = 0;
        ku = 0;
    }
    else if (uplo == Uplo::Lower) {
        kl = std::max<idx_t>(n, 1) - 1;
        ku = 0;
    }
    else {
        kl = 0;
        ku = std::max<idx_t>(n, 1) - 1;
    }

    if (imat == 3 or imat == 9)
        cndnum = badc1;
    else if (imat == 4 or imat == 10)
        cndnum = badc2;
    else
        cndnum = real_t(2);

    if (imat == 5)
        anorm = smlnum;
    else if (imat == 6)
        anorm = bignum;
    else
        anorm = one;

    std::vector<T> d(n);

    generate_with_known_sv(d, 3, cndnum, anorm, A, kl, ku);
}

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
    idx_t n = GENERATE(4, 5, 8, 20);
    Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);
    Op trans = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);
    // int imat = GENERATE(1, 2, 3, 4, 6);
    int imat = GENERATE(1, 2, 3, 4, 5, 6);
    Diag diag = Diag::NonUnit;

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

    generate_latrs_test_matrix(imat, uplo, A, b, diag);

    for (idx_t i = 0; i < n; ++i)
        b[i] = rand_helper<T>();

    for (idx_t i = 0; i < n; ++i)
        x[i] = b[i];

    // Calculate solution with trsv for testing purposes
    // trsv(uplo, trans, diag, A, x);
    // for (idx_t i = 0; i < n; ++i)
    //     std::cout << x[i] << ", ";
    // std::cout << std::endl;
    // for (idx_t i = 0; i < n; ++i)
    //     x[i] = b[i];

    DYNAMIC_SECTION("n = " << n << " Uplo = " << uplo << " Trans = " << trans
                           << " diag = " << diag << " imat = " << imat)
    {
        latrs(uplo, trans, diag, false, A, x, scale, cnorm);

        trmv(uplo, trans, diag, A, x);

        auto itemp = iamax(x);
        real_t xnorm = tlapack::abs1(x[itemp]);
        real_t enorm = real_t(0);
        for (idx_t i = 0; i < n; ++i)
            enorm += abs1(x[i] - scale * b[i]);

        CHECK(enorm <= tol * xnorm);
    }
}
