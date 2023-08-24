/// @file test_getrf.cpp
/// @author Ali Lotfi, University of Colorado Denver, USA
/// @brief Test the LU factorization of a matrix A
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
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
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/lu_mult.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("LU factorization of a general m-by-n matrix",
                   "[getrf]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // m and n represent no. rows and columns of the matrices we will be testing
    // respectively
    idx_t m = GENERATE(10, 20, 30);
    idx_t n = GENERATE(10, 20, 30);
    GetrfVariant variant =
        GENERATE(GetrfVariant::Level0, GetrfVariant::Recursive);

    DYNAMIC_SECTION("m = " << m << " n = " << n
                           << " variant = " << (char)variant)
    {
        idx_t k = min<idx_t>(m, n);

        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(max(m, n)) * eps;

        // Initialize matrices A, and A_copy to run tests on
        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> A_copy_;
        auto A_copy = new_matrix(A_copy_, m, n);

        // Update A with random numbers
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i) {
                // A(i, j) = rand_helper<T>();
                A(i, j) = rand_helper<T>();
            }

        // We will make a deep copy A
        // We intend to test A=LU, however, since after calling getrf, A will be
        // udpated then to test A=LU, we'll make a deep copy of A prior to
        // calling getrf
        lacpy(GENERAL, A, A_copy);

        real_t norma = tlapack::lange(tlapack::MAX_NORM, A);
        // Initialize piv vector to all zeros
        std::vector<idx_t> piv(k, idx_t(0));
        // Run getrf and both A and piv will be update
        getrf(A, piv, GetrfOpts{variant});

        // A contains L and U now, then form A <--- LU
        if (m > n) {
            auto A0 = tlapack::slice(A, range(0, n), range(0, n));
            auto A1 = tlapack::slice(A, range(n, m), range(0, n));
            trmm(RIGHT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1),
                 A0, A1);
            lu_mult(A0);
        }
        else if (m < n) {
            auto A0 = tlapack::slice(A, range(0, m), range(0, m));
            auto A1 = tlapack::slice(A, range(0, m), range(m, n));
            trmm(LEFT_SIDE, LOWER_TRIANGLE, NO_TRANS, UNIT_DIAG, real_t(1), A0,
                 A1);
            lu_mult(A0);
        }
        else
            lu_mult(A);

        // Now that piv is updated, we work our way backwards in piv and switch
        // rows of LU
        for (idx_t j = k - idx_t(1); j != idx_t(-1); j--) {
            auto vect1 = tlapack::row(A, j);
            auto vect2 = tlapack::row(A, piv[j]);
            tlapack::swap(vect1, vect2);
        }

        // A <- A_original - LU
        for (idx_t i = 0; i < m; i++)
            for (idx_t j = 0; j < n; j++)
                A(i, j) -= A_copy(i, j);

        // Check for relative error: norm(A-LU)/norm(A)
        real_t error = tlapack::lange(tlapack::MAX_NORM, A) / norma;
        CHECK(error <= tol);
    }
}
