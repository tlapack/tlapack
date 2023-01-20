/// @file test_potrf.cpp Test the Cholesky factorization of a symmetric positive
/// definite matrix
/// @author Weslley S Pereira, University of Colorado Denver, USA
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
#include <tlapack/lapack/lanhe.hpp>

// Other routines
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/potrf.hpp>

#include "TestUploMatrix.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "Cholesky factorization of a Hermitian positive-definite matrix",
    "[potrf]",
    TLAPACK_TYPES_TO_TEST)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    using variant_t = std::pair<PotrfVariant, idx_t>;
    const variant_t variant = GENERATE((variant_t(PotrfVariant::Blocked, 1)),
                                       (variant_t(PotrfVariant::Blocked, 2)),
                                       (variant_t(PotrfVariant::Blocked, 7)),
                                       (variant_t(PotrfVariant::Blocked, 10)),
                                       (variant_t(PotrfVariant::Recursive, 0)));
    const idx_t n = GENERATE(10, 19, 30);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    INFO("Variant: " << (char)variant.first << ", nb: " << variant.second);
    INFO("Matrix size: " << n << "x" << n);
    INFO("Uplo: " << uplo);

    // eps is the machine precision, and tol is the tolerance we accept for
    // tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(n) * eps;

    // Create matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> L_;
    auto L = new_matrix(L_, n, n);
    std::vector<T> E_;
    auto E = new_matrix(E_, n, n);

    // Update A with random numbers
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if (uplo == Uplo::Lower && i >= j)
                A(i, j) = rand_helper<T>();
            else if (uplo == Uplo::Upper && i <= j)
                A(i, j) = rand_helper<T>();
            else
                A(i, j) = real_t(0xCAFEBABE);
        }
        A(j, j) += real_t(n);
    }

    lacpy(dense, A, L);
    real_t normA = tlapack::lanhe(tlapack::Norm::Max, uplo, A);

    // Run the Cholesky factorization
    potrf_opts_t<idx_t> opts;
    opts.variant = variant.first;
    opts.nb = variant.second;
    int info = potrf(uplo, L, opts);

    // Check that the factorization was successful
    REQUIRE(info == 0);

    // Initialize E with the hermitian part of L
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i) {
            if (uplo == Uplo::Lower && i <= j)
                E(i, j) = conj(L(j, i));
            else if (uplo == Uplo::Upper && i >= j)
                E(i, j) = conj(L(j, i));
            else
                E(i, j) = real_t(0);
        }

    // Compute E = L*L^H or E = L^H*L
    if (uplo == Uplo::Lower)
        trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, real_t(1), L,
             E);
    else
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, real_t(1), L,
             E);

    // Check that the factorization is correct
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++) {
            if (uplo == Uplo::Lower && i >= j)
                E(i, j) -= A(i, j);
            else if (uplo == Uplo::Upper && i <= j)
                E(i, j) -= A(i, j);
        }

    // Check for relative error: norm(A-cholesky(A))/norm(A)
    real_t error = tlapack::lanhe(tlapack::Norm::Max, uplo, E) / normA;
    CHECK(error <= tol);
}

TEMPLATE_TEST_CASE("Cholesky factorization access valid positions only",
                   "[potrf]",
                   TLAPACK_LEGACY_REAL_TYPES_TO_TEST)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    const Layout L = layout<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    using variant_t = std::pair<PotrfVariant, idx_t>;
    const variant_t variant = GENERATE((variant_t(PotrfVariant::Blocked, 2)),
                                       (variant_t(PotrfVariant::Recursive, 0)));
    const idx_t n = GENERATE(10);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    INFO("Variant: " << (char)variant.first << ", nb: " << variant.second);
    INFO("Matrix size: " << n << "x" << n);
    INFO("Uplo: " << uplo);

    // Create matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);

    // Update A with random numbers
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if (uplo == Uplo::Lower && i >= j)
                A(i, j) = rand_helper<T>();
            else if (uplo == Uplo::Upper && i <= j)
                A(i, j) = rand_helper<T>();
            else
                A(i, j) = real_t(0xCAFEBABE);
        }
        A(j, j) += real_t(n);
    }

    // Run the Cholesky factorization
    potrf_opts_t<idx_t> opts;
    opts.variant = variant.first;
    opts.nb = variant.second;
    if (uplo == Uplo::Lower) {
        TestUploMatrix<T, idx_t, Uplo::Lower, L> testA(A);

        int info = potrf(uplo, testA, opts);
        REQUIRE(info == 0);
    }
    else {
        TestUploMatrix<T, idx_t, Uplo::Upper, L> testA(A);

        int info = potrf(uplo, testA, opts);
        REQUIRE(info == 0);
    }
}
