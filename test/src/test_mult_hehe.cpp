/// @file test_mult_hehe.cpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
/// @brief Test Hermitian multiplication
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
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/mult_hehe.hpp>

using namespace tlapack;

// Helper to set alpha and beta safely for both real and complex types
template <typename T>
void setScalar(T& alpha, real_type<T> a_real, real_type<T> a_imag)
{
    alpha = a_real;
}

template <typename T>
void setScalar(std::complex<T>& alpha, real_type<T> a_real, real_type<T> a_imag)
{
    alpha.real(a_real);
    alpha.imag(a_imag);
}

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

    const idx_t n = GENERATE(1, 3, 5, 9);

    const Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);

    T alpha, beta;

    real_t a_real = GENERATE(real_t(2), real_t(-5), real_t(-2.4), real_t(8.6));
    real_t a_imag = GENERATE(real_t(2), real_t(-5), real_t(-2.4), real_t(8.6));
    real_t b_real = GENERATE(real_t(2), real_t(-5), real_t(-2.4), real_t(6.5));
    real_t b_imag = GENERATE(real_t(2), real_t(-5), real_t(-2.4), real_t(6.5));

    setScalar(alpha, a_real, b_real);
    setScalar(beta, b_real, b_imag);

    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    DYNAMIC_SECTION("n = " << n << " alpha = " << alpha << " beta = " << beta
                           << " Uplo" << uplo)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        std::vector<T> C_;
        auto C = new_matrix(C_, n, n);
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        std::vector<T> B_;
        auto B = new_matrix(B_, n, n);
        std::vector<T> F_;
        auto F = new_matrix(F_, n, n);
        std::vector<T> D_;
        auto D = new_matrix(D_, n, n);
        std::vector<T> E_;
        auto E = new_matrix(E_, n, n);

        // Generate n-by-n random matrix
        mm.random(A);
        mm.random(B);
        mm.random(C);

        for (idx_t i = 0; i < n; i++) {
            A(i, i) = real(A(i, i));
            B(i, i) = real(B(i, i));
        }

        lacpy(GENERAL, A, D);
        lacpy(GENERAL, B, E);
        lacpy(GENERAL, C, F);
        if (uplo == Uplo::Upper) {
            for (idx_t i = 0; i < n; i++)
                for (idx_t j = 0; j < i; ++j) {
                    D(i, j) = conj(D(j, i));
                    E(i, j) = conj(E(j, i));
                    A(i, j) = 0;
                    B(i, j) = 0;
                }
        }
        else {
            for (idx_t i = 0; i < n; i++)
                for (idx_t j = i + 1; j < n; ++j) {
                    D(i, j) = conj(D(j, i));
                    E(i, j) = conj(E(j, i));
                    A(i, j) = 0;
                    B(i, j) = 0;
                }
        }

        gemm(Op::NoTrans, Op::NoTrans, alpha, D, E, beta, F);

        real_t normF = lange(FROB_NORM, F);

        mult_hehe(uplo, alpha, A, B, beta, C);

        for (idx_t i = 0; i < n; ++i)
            for (idx_t j = 0; j < n; ++j) {
                C(i, j) -= F(i, j);
            }

        real_t normC = lange(FROB_NORM, C);

        normC = normC / normF;

        // Check if residual is 0 with machine accuracy
        CHECK(normC <= tol);
        if (uplo == Uplo::Upper) {
            real_t sum(0);
            for (idx_t j = 0; j < n; j++)
                for (idx_t i = j + 1; i < n; i++)
                    sum += abs1(A(i, j) - B(i, j));
            CHECK(sum == real_t(0));
        }
        else {
            real_t sum(0);
            for (idx_t j = 0; j < n; j++)
                for (idx_t i = 0; i < j; i++)
                    sum += abs1(A(i, j) - B(i, j));
            CHECK(sum == real_t(0));
        }
    }
}
