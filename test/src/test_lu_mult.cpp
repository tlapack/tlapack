/// @file test_lu_mult.cpp
/// @author Lindsay Slager, University of Colorado Denver, USA
/// @brief Test LU multiplication
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
#include <tlapack/lapack/lu_mult.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("lu multiplication is backward stable",
                   "[lu check][lu][qrt]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    idx_t n, nx;

    n = GENERATE(1, 2, 6, 9);
    nx = GENERATE(1, 2, 4, 5);

    DYNAMIC_SECTION("n = " << n << " nx = " << nx)
    {
        if (nx <= n) {
            const real_t eps = ulp<real_t>();
            const real_t tol = real_t(n) * eps;

            std::vector<T> L_;
            auto L = new_matrix(L_, n, n);
            std::vector<T> U_;
            auto U = new_matrix(U_, n, n);
            std::vector<T> A_;
            auto A = new_matrix(A_, n, n);

            // Generate n-by-n random matrix
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < n; ++i)
                    A(i, j) = rand_helper<T>();

            lacpy(Uplo::Lower, A, L);
            laset(Uplo::Upper, real_t(0), real_t(1), L);

            laset(Uplo::Lower, real_t(0), real_t(0), U);
            lacpy(Uplo::Upper, A, U);

            real_t norma = lange(MAX_NORM, A);

            {
                lu_mult(A);

                // Calculate residual

                gemm(Op::NoTrans, Op::NoTrans, real_t(1), L, U, real_t(-1), A);

                real_t lu_mult_res_norm = lange(MAX_NORM, A) / norma;
                CHECK(lu_mult_res_norm <= tol);
            }
        }
    }
}
