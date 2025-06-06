/// @file test_lu_mult.cpp
/// @author Lindsay Slager, University of Colorado Denver, USA
/// @brief Test LU multiplication
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
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lantr.hpp>
// Other routines
#include <tlapack/blas/gemmtr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("check for gemmtr multiplication",
                   "[gemmtr]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n;

    idx_t k;

    // n = GENERATE(1, 2, 6, 9);

    // k = GENERATE(3, 5, 7, 8);
    n = GENERATE(9);

    k = GENERATE(5);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);
    const Op transA = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);
    const Op transB = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);
    DYNAMIC_SECTION("n = " << n << " k = " << k)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n + k) * eps;

        // Generating Matrices A, B, C0, C1, C2
        std::vector<T> A_;
        auto A = new_matrix(A_, n, k);
        std::vector<T> B_;
        auto B = new_matrix(B_, k, n);
        std::vector<T> C0_;
        auto C0 = new_matrix(C0_, n, n);
        std::vector<T> C1_;
        auto C1 = new_matrix(C1_, n, n);
        std::vector<T> C2_;
        auto C2 = new_matrix(C2_, n, n);

        // Generate n-by-n random matrix
        mm.random(A);
        mm.random(B);
        mm.random(C0);

        lacpy(GENERAL, C0, C1);
        lacpy(GENERAL, C0, C2);

        real_t norma = lange(MAX_NORM, A);
        real_t normb = lange(MAX_NORM, B);

        T alpha, beta;

        if constexpr (is_complex<T>) {
            alpha = T(-2, 5);
            beta = T(-7, 4);
        }
        else {
            alpha = T(-3);
            beta = T(8);
        }
        {
            // Calculate residual
            // Upper, no trans, no trans DONE

            // Uplo uplo = Uplo::Lower;
            // Op transA = Op::ConjTrans;
            // Op transB = Op::ConjTrans;

            // Lower, no trans, no trans, IN PROGRESS

            real_t normc = lantr(MAX_NORM, uplo, NON_UNIT_DIAG, C0);

            gemmtr(uplo, transA, transB, alpha, A, B, beta, C1);

            gemm(transA, transB, alpha, A, B, beta, C2);

            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n;
                     j++)  // Check of upper part. iterates to right of row
                    for (idx_t i = 0; i <= j;
                         i++)  // Iterates Down columns and touches diagonals
                        C1(i, j) -= C2(i, j);

                real_t normres =
                    lantr(MAX_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, C1);
                CHECK(normres <=
                      tol * (abs1(alpha) * norma * normb + abs1(beta) * normc));

                real_t sum = 0;
                for (idx_t j = 0; j < n; j++)  // Check strictly lower part
                    for (idx_t i = j + 1; i < n; i++)  //
                        sum += abs1(
                            C1(i, j) -
                            C0(i, j));  // Subtracts all of lower elements
                                        // element wise then puts them in sum

                CHECK(sum == real_t(0));  // Sum should be exactly 0 since
                                          // elements untouched
            }
            else {
                for (idx_t i = 0; i < n; i++)       // Check of lower part
                    for (idx_t j = 0; j <= i; j++)  //  Touches diagonals
                        C1(i, j) -= C2(i, j);

                real_t normres =
                    lantr(MAX_NORM, LOWER_TRIANGLE, NON_UNIT_DIAG, C1);
                CHECK(normres <=
                      tol * (abs1(alpha) * norma * normb + abs1(beta) * normc));

                real_t sum = 0;
                for (idx_t i = 0; i < n; i++)  // Check strictly upper part
                    for (idx_t j = i + 1; j < n;
                         j++)  // Does not touch diagonals
                        sum += abs1(
                            C1(i, j) -
                            C0(i, j));  // Subtracts all of upper elements
                                        // element wise then puts them in sum

                CHECK(sum == real_t(0));  // Sum should be exactly 0 since
                                          // elements untouched
            }
        }
    }
}
