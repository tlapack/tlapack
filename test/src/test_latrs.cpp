/// @file test_latrs.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test safe scaling linear solver.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

// Other routines
#include <tlapack/blas/trmv.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/latrs.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Internal safe scaling solver works",
                   "[latrs]",
                   TLAPACK_TYPES_TO_TEST)
{
    //
    // In this test, we don't use latrs, but we use latrs_solve_scaled_system
    // with systems where TRSV should work without scaling.
    // We need to use latrs_solve_scaled_system directly because latrs
    // should always call TRSV in this case, and we want to test the
    // scaling solver.
    //

    using matrix_t = TestType;
    using vector_t = vector_type<TestType>;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<TA>;
    using range = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // Functor
    Create<vector_t> new_vector;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t n = GENERATE(1, 2, 3, 4, 5, 10);
    const Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);
    const Diag diag = GENERATE(Diag::NonUnit, Diag::Unit);
    const Op trans = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);
    const real_t zero(0);
    const real_t one(1);

    const real_t tol = real_t(5) * n * uroundoff<real_t>();

    std::vector<TA> A_;
    auto A = new_matrix(A_, n, n);
    mm.randn(A);
    // Make A triangular by taking LU factorization and zeroing out the other
    // triangle
    std::vector<idx_t> piv_;
    auto piv = new_vector(piv_, n);
    getrf(A, piv);

    // Zero out the opposite triangle
    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = j + 1; i < n; ++i) {
                A(i, j) = zero;
            }
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < j; ++i) {
                A(i, j) = zero;
            }
        }
    }

    std::vector<TA> x_true;
    auto x_t = new_vector(x_true, n);
    for (idx_t i = 0; i < n; ++i) {
        x_t[i] = rand_helper<TA>(mm.gen);
    }

    // Compute b = A * x_true
    std::vector<TA> b_;
    auto b = new_vector(b_, n);
    for (idx_t i = 0; i < n; ++i) {
        b[i] = x_t[i];
    }
    trmv(uplo, trans, diag, A, b);

    // Make a copy of b
    std::vector<TA> x_;
    auto x = new_vector(x_, n);
    for (idx_t i = 0; i < n; ++i) {
        x[i] = b[i];
    }

    DYNAMIC_SECTION(" n = " << n << " uplo = " << char(uplo) << " diag = "
                            << char(diag) << " trans = " << char(trans))
    {
        real_t scale;
        std::vector<real_t> cnorm_(n);
        auto cnorm = new_vector(cnorm_, n);

        // Use latrs_calc_cnorm to compute column norms
        real_t tscal;
        char normin = 'N';
        int info_cnorm =
            latrs_calc_cnorm(uplo, trans, diag, normin, A, cnorm, tscal);
        CHECK(info_cnorm == 0);
        CHECK(tscal == real_t(1));

        idx_t j = iamax(x);
        real_t xmax = abs1(x[j]);

        latrs_solve_scaled_system(uplo, trans, diag, A, x, scale, cnorm, tscal,
                                  xmax);

        // Check that A * x = scale * b
        std::vector<TA> Ax_;
        auto Ax = new_vector(Ax_, n);
        for (idx_t i = 0; i < n; ++i) {
            Ax[i] = x[i];
        }
        trmv(uplo, trans, diag, A, Ax);

        real_t x_true_norm = asum(x_t);
        real_t anorm = lange(Norm::One, A);
        for (idx_t i = 0; i < n; ++i) {
            CHECK(abs(Ax[i] - scale * b[i]) <= tol * x_true_norm * anorm);
        }
    }
}

TEMPLATE_TEST_CASE("latrs works for systems that don't need scaling",
                   "[latrs]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using vector_t = vector_type<TestType>;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<TA>;
    using range = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // Functor
    Create<vector_t> new_vector;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t n = GENERATE(1, 2, 3, 4, 5, 10);
    const Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);
    const Diag diag = GENERATE(Diag::NonUnit, Diag::Unit);
    const Op trans = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);
    const real_t zero(0);
    const real_t one(1);

    const real_t tol = real_t(5) * real_t(n) * uroundoff<real_t>();

    std::vector<TA> A_;
    auto A = new_matrix(A_, n, n);
    mm.randn(A);
    // Make A triangular by taking LU factorization and zeroing out the other
    // triangle
    std::vector<idx_t> piv_;
    auto piv = new_vector(piv_, n);
    getrf(A, piv);

    // Zero out the opposite triangle
    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = j + 1; i < n; ++i) {
                A(i, j) = zero;
            }
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < j; ++i) {
                A(i, j) = zero;
            }
        }
    }

    std::vector<TA> x_true;
    auto x_t = new_vector(x_true, n);
    for (idx_t i = 0; i < n; ++i) {
        x_t[i] = rand_helper<TA>(mm.gen);
    }

    // Compute b = A * x_true
    std::vector<TA> b_;
    auto b = new_vector(b_, n);
    for (idx_t i = 0; i < n; ++i) {
        b[i] = x_t[i];
    }
    trmv(uplo, trans, diag, A, b);

    // Make a copy of b
    std::vector<TA> x_;
    auto x = new_vector(x_, n);
    for (idx_t i = 0; i < n; ++i) {
        x[i] = b[i];
    }

    DYNAMIC_SECTION(" n = " << n << " uplo = " << char(uplo) << " diag = "
                            << char(diag) << " trans = " << char(trans))
    {
        real_t scale;
        std::vector<real_t> cnorm_(n);
        auto cnorm = new_vector(cnorm_, n);

        char normin = 'N';
        int info = latrs(uplo, trans, diag, normin, A, x, scale, cnorm);
        CHECK(info == 0);

        // Check that A * x = scale * b
        std::vector<TA> Ax_;
        auto Ax = new_vector(Ax_, n);
        for (idx_t i = 0; i < n; ++i) {
            Ax[i] = x[i];
        }
        trmv(uplo, trans, diag, A, Ax);

        real_t x_true_norm = asum(x_t);
        real_t anorm = lange(Norm::One, A);
        for (idx_t i = 0; i < n; ++i) {
            CHECK(abs(Ax[i] - scale * b[i]) <= tol * x_true_norm * anorm);
        }
    }
}
