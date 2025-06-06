/// @file test_mult_uhu.cpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
/// @brief Test UHU multiplication
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/lantr.hpp>
#include <tlapack/lapack/mult_uhu.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("uhu multiplication is backward stable",
                   "[uhu check]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;
    using range = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n, nx;

    n = GENERATE(1, 2, 6, 9);
    nx = GENERATE(1, 2, 4, 5);

    DYNAMIC_SECTION("n = " << n << " nx = " << nx)
    {
        if (nx <= n) {
            const real_t eps = ulp<real_t>();
            const real_t tol = real_t(n) * eps;

            std::vector<T> C_;
            auto C = new_matrix(C_, n, n);
            std::vector<T> A_;
            auto A = new_matrix(A_, n, n);
            std::vector<T> B_;
            auto B = new_matrix(B_, n, n);

            // Generate n-by-n random matrix
            mm.random(A);

            lacpy(GENERAL, A, C);
            lacpy(GENERAL, A, B);

            auto subA = slice(A, range(1, n), range(0, n - 1));
            laset(LOWER_TRIANGLE, real_t(0), real_t(0), subA);

            real_t normA = lantr(MAX_NORM, UPPER_TRIANGLE, Diag::NonUnit, A);

            {
                mult_uhu_Opts opts;

                opts.nx = nx;
                // A = C^H * C
                mult_uhu(C, opts);

                // C = A^H*A - C
                herk(UPPER_TRIANGLE, Op::ConjTrans, real_t(1), A, real_t(-1),
                     C);

                // Check if residual is 0 with machine accuracy
                real_t uhu_mult_res_norm =
                    lantr(MAX_NORM, UPPER_TRIANGLE, Diag::NonUnit, C);
                CHECK(uhu_mult_res_norm <= tol * normA * normA);

                real_t sum(0);
                for (idx_t j = 0; j < n; j++)
                    for (idx_t i = j + 1; i < n; i++)
                        sum += abs1(C(i, j) - B(i, j));
                CHECK(sum == real_t(0));
            }
        }
    }
}
