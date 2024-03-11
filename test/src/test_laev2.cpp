/// @file test_laev2.cpp Test the solution of 2x2 symmetric eigenvalue
/// problems
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Other routines
#include <tlapack/blas/rotg.hpp>
#include <tlapack/lapack/lae2.hpp>
#include <tlapack/lapack/laev2.hpp>
#include <tlapack/lapack/lapy2.hpp>
#include <tlapack/lapack/singularvalues22.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("check that laev2 gives correct eigenvalues",
                   "[symmetriceigenvalues]",
                   float,
                   double)
{
    using T = TestType;

    // MatrixMarket reader
    uint64_t seed = GENERATE(1, 2, 3, 4, 5, 6);

    rand_generator gen;
    gen.seed(seed);

    const real_type<T> eps = ulp<real_type<T>>();

    DYNAMIC_SECTION("seed = " << seed)
    {
        // Matrix A = [a b; b c]
        T a = rand_helper<T>(gen);
        T b = rand_helper<T>(gen);
        T c = rand_helper<T>(gen);

        // Compute eigenvalues
        T s1, s2, cs, sn;
        laev2(a, b, c, s1, s2, cs, sn);
        T Anorm = lapy2(lapy2(a, b), lapy2(b, c));

        // Check backward error
        {
            // Calculate B = [cs -sn; sn cs] * [s1 0; 0 s2] * [cs sn; -sn cs]
            T b11 = cs * cs * s1 + sn * sn * s2;
            T b12 = cs * sn * (s1 - s2);
            T b22 = sn * sn * s1 + cs * cs * s2;

            CHECK(abs(b11 - a) <= 1.0e1 * eps * Anorm);
            CHECK(abs(b12 - b) <= 1.0e1 * eps * Anorm);
            CHECK(abs(b22 - c) <= 1.0e1 * eps * Anorm);
        }

        // Check first eigenvalue
        // We check that the matrix B = A - s1 * I is singular
        {
            T b11 = a - s1;
            T b12 = b;
            T b21 = b;
            T b22 = c - s1;

            // Make B upper triangular using givens rotation
            T cs, sn;
            rotg(b11, b21, cs, sn);
            T temp = cs * b12 + sn * b22;
            b22 = -sn * b12 + cs * b22;
            b12 = temp;

            // Calculate singular values of B
            T ssmin, ssmax;
            singularvalues22(b11, b12, b22, ssmin, ssmax);

            // Check that ssmin is small enough
            CHECK(ssmin <= 1.0e1 * eps * Anorm);
        }

        // Check second eigenvalue
        // We check that the matrix B = A - s2 * I is singular
        {
            T b11 = a - s2;
            T b12 = b;
            T b21 = b;
            T b22 = c - s2;

            // Make B upper triangular using givens rotation
            T cs, sn;
            rotg(b11, b21, cs, sn);
            T temp = cs * b12 + sn * b22;
            b22 = -sn * b12 + cs * b22;
            b12 = temp;

            // Calculate singular values of B
            T ssmin, ssmax;
            singularvalues22(b11, b12, b22, ssmin, ssmax);

            // Check that ssmin is small enough
            CHECK(ssmin <= 1.0e1 * eps * Anorm);
        }
    }
}