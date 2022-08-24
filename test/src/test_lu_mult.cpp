/// @file test_lu_mult.cpp
/// @brief Test LU multiplication
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("lu multiplication is backward stable", "[lu check][lu][qrt]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    idx_t n, nx;

    n = GENERATE(1, 2, 6, 9);
    nx = GENERATE(1, 2, 4, 5);

    if(nx <= n){

        const real_t eps = ulp<real_t>();
        const real_t tol = n * eps;

        std::unique_ptr<T[]> L_(new T[n * n]);
        std::unique_ptr<T[]> U_(new T[n * n]);
        std::unique_ptr<T[]> A_(new T[n * n]);

        auto L = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &L_[0], n);
        auto U = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &U_[0], n);
        auto A = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &A_[0], n);

        // Generate n-by-n random matrix
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        lacpy(Uplo::Lower, A, L);
        laset(Uplo::Upper, real_t(0), real_t(1), L);

        laset(Uplo::Lower, real_t(0), real_t(0), U);
        lacpy(Uplo::Upper, A, U);

        real_t norma = lange(max_norm, A);

        DYNAMIC_SECTION("n = " << n)
        {
            lu_mult(A);

            // Calculate residual

            gemm(Op::NoTrans, Op::NoTrans, real_t(1), L, U, real_t(-1), A);

            real_t lu_mult_res_norm = lange(max_norm, A) / norma;
            CHECK(lu_mult_res_norm <= tol);
        }
    }
}