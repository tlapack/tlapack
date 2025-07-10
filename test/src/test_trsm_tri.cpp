/// @file test_trmm_out.cpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
/// @brief Test out-of-place triangular matrix-matrix multiplication.
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

// Other routines

#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/trmm_out.hpp>
#include <tlapack/lapack/trsm_tri.hpp>

using namespace tlapack;

template <typename T>
void setScalar(T& alpha, real_type<T> aReal, real_type<T> aImag)
{
    alpha = aReal;
}

template <typename T>
void setScalar(std::complex<T>& alpha, real_type<T> aReal, real_type<T> aImag)
{
    alpha.real(aReal);
    alpha.imag(aImag);
}

TEMPLATE_TEST_CASE("triagular matrix-matrix multiplication is backward stable",
                   "[triangular matrix-matrix check]",
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

    const idx_t n = GENERATE(1, 3, 5, 7, 9);

    srand(3);

    T alpha;

    srand(3);

    // Random number engine (seed with a random device)
    std::random_device rd;
    std::mt19937 gen(rd());

    // Uniform distribution: 0 or 1
    std::uniform_int_distribution<> dist(0, 1);

    // Generate either -1 or 1
    float valueA = dist(gen) == 0 ? -1.0 : 1.0;
    float valueB = dist(gen) == 0 ? -1.0 : 1.0;

    real_t aReal = real_t(valueA * (float)rand() / (float)RAND_MAX);
    real_t aImag = real_t(valueB * (float)rand() / (float)RAND_MAX);

    setScalar(alpha, aReal, aImag);

    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    const Side sideA = GENERATE(Side::Left, Side::Right);

    const Op transA = GENERATE(Op::NoTrans, Op::ConjTrans, Op::Trans);

    const Diag diagA = GENERATE(Diag::NonUnit, Diag::Unit);

    DYNAMIC_SECTION("n = " << n << " alpha = " << alpha << " Uplo = " << uplo
                           << " transA = " << transA << " diagA = " << diagA)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        Uplo uploA;
        if (transA != tlapack::Op::NoTrans)
            uploA = (uplo == tlapack::Uplo::Upper) ? tlapack::Uplo::Lower
                                                   : tlapack::Uplo::Upper;
        else
            uploA = uplo;

        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        std::vector<T> A_orig_;
        auto A_orig = new_matrix(A_orig_, n, n);
        std::vector<T> B_;
        auto B = new_matrix(B_, n, n);

        mm.random(A);
        mm.random(B);

        lacpy(GENERAL, A, A_orig);

        std::vector<T> X_;
        auto X = new_matrix(X_, n, n);
        lacpy(GENERAL, B, X);

        if (transA == tlapack::Op::NoTrans) {
            if (uplo == tlapack::Uplo::Lower) {
                trsm_tri(sideA, uplo, transA, diagA, alpha, A, X);

                // check to make sure we did not touch the upper portion
                real_t sum(0);
                for (idx_t j = 0; j < n; j++)
                    for (idx_t i = 0; i < j; i++)
                        sum += abs1(X(i, j) - B(i, j));
                CHECK(sum == real_t(0));

                // zero out the upper part of A
                auto tempA = slice(A, range(0, n - 1), range(1, n));
                laset(tlapack::Uplo::Upper, real_t(0), real_t(0), tempA);

                // zero out the upper part of B
                auto tempB = slice(B, range(0, n - 1), range(1, n));
                laset(tlapack::Uplo::Upper, real_t(0), real_t(0), tempB);

                // zero out the upper part of X
                auto tempX = slice(X, range(0, n - 1), range(1, n));
                laset(tlapack::Uplo::Upper, real_t(0), real_t(0), tempX);
            }
            else {
                trsm_tri(sideA, uplo, transA, diagA, alpha, A, X);

                // check to make sure we did not touch the lower portion
                real_t sum(0);
                for (idx_t j = 0; j < n; j++)
                    for (idx_t i = j + 1; i < n; i++)
                        sum += abs1(X(i, j) - B(i, j));
                CHECK(sum == real_t(0));

                // zero out the lower part of A
                auto tempA = slice(A, range(1, n), range(0, n - 1));
                laset(tlapack::Uplo::Lower, real_t(0), real_t(0), tempA);

                // zero out the lower part of B
                auto tempB = slice(B, range(1, n), range(0, n - 1));
                laset(tlapack::Uplo::Lower, real_t(0), real_t(0), tempB);

                // zero out the lower part of X
                auto tempX = slice(X, range(1, n), range(0, n - 1));
                laset(tlapack::Uplo::Lower, real_t(0), real_t(0), tempX);
            }
        }
        else {
            if (uplo == tlapack::Uplo::Lower) {
                trsm_tri(sideA, uplo, transA, diagA, alpha, A, X);

                // check to make sure we did not touch the upper portion
                real_t sum(0);
                for (idx_t j = 0; j < n; j++)
                    for (idx_t i = 0; i < j; i++)
                        sum += abs1(X(i, j) - B(i, j));
                CHECK(sum == real_t(0));

                // zero out the lower part of A
                auto tempA = slice(A, range(1, n), range(0, n - 1));
                laset(tlapack::Uplo::Lower, real_t(0), real_t(0), tempA);

                // zero out the upper part of B
                auto tempB = slice(B, range(0, n - 1), range(1, n));
                laset(tlapack::Uplo::Upper, real_t(0), real_t(0), tempB);

                // zero out the upper part of X
                auto tempX = slice(X, range(0, n - 1), range(1, n));
                laset(tlapack::Uplo::Upper, real_t(0), real_t(0), tempX);
            }
            else {
                // solve the linear system
                trsm_tri(sideA, uplo, transA, diagA, alpha, A, X);

                // check to make sure we did not touch the lower portion
                real_t sum(0);
                for (idx_t j = 0; j < n; j++)
                    for (idx_t i = j + 1; i < n; i++)
                        sum += abs1(X(i, j) - B(i, j));
                CHECK(sum == real_t(0));

                // zero out the upper part of A
                auto tempA = slice(A, range(0, n - 1), range(1, n));
                laset(tlapack::Uplo::Upper, real_t(0), real_t(0), tempA);

                // zero out the lower part of B
                auto tempB = slice(B, range(1, n), range(0, n - 1));
                laset(tlapack::Uplo::Lower, real_t(0), real_t(0), tempB);

                // zero out the lower part of X
                auto tempX = slice(X, range(1, n), range(0, n - 1));
                laset(tlapack::Uplo::Lower, real_t(0), real_t(0), tempX);
            }
        }

        real_t normA = lange(Norm::Fro, A);
        real_t normB = lange(Norm::Fro, B);
        real_t normX = lange(Norm::Fro, X);

        // Check
        trmm(sideA, uploA, transA, diagA, real_t(1), A, X);
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < n; j++) {
                X(i, j) -= alpha * B(i, j);
            }
        }

        // Compute norm of the residual
        real_t normRes = lange(Norm::Fro, X);

        CHECK(normRes <= tol * (abs1(alpha) * normB + normA * normX));
    }
}