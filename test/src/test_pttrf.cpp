/// @file test_pttrf.cpp Test the Cholesky factorization of a symmetric positive
/// definite tridiagonal matrix
/// @author Hugh M. Kadhem, University of California Berkeley, USA
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
#include <tlapack/lapack/lanhe.hpp>

// Other routines
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/pttrf.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "Cholesky factorization of a Hermitian positive-definite tridiagonal "
    "matrix",
    "[pttrf]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t n = GENERATE(10, 19, 30);

    DYNAMIC_SECTION("n = " << n)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        // Create matrices
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        std::vector<T> B_;
        auto B = new_matrix(B_, n, n);

        // Update A with random numbers, and make it positive definite
        mm.random(Uplo::Lower, A);
        for (idx_t i = 0; i < n; ++i) {
            for (idx_t j = 0; j < n; ++j) {
                if (i == j) {
                    A(j, j) = abs(A(j, j)) + real_t(n);
                }
                else if (i != j + 1) {
                    A(i, j) = T(0);
                }
            }
        }

        lacpy(GENERAL, A, B);
        auto D = diag(B, 0);
        auto E = diag(B, -1);
        real_t normA = tlapack::lanhe(tlapack::MAX_NORM, Uplo::Lower, A);

        // Run the Cholesky factorization
        int info = pttrf(D, E);

        // Check that the factorization was successful
        REQUIRE(info == 0);

        real_t absErr = abs(A(0, 0) - D[0]);
        for (idx_t j = 0; j < n - 1; ++j) {
            absErr = max(absErr, abs(A(j + 1, j) - D[j] * E[j]));
            absErr = max(absErr, abs(A(j + 1, j + 1) -
                                     D[j] * E[j] * conj(E[j]) - D[j + 1]));
        }

        // Check for relative error: norm(A-cholesky(A))/norm(A)
        real_t error = absErr / normA;
        CHECK(error <= tol);
    }
}
