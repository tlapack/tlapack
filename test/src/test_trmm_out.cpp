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
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lantr.hpp>
#include <tlapack/lapack/trmm_out.hpp>

using namespace tlapack;

// Helper to set alpha and beta safely for both real and complex types
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

    const idx_t n = GENERATE(3, 5, 7, 9);
    const idx_t m = GENERATE(2, 4, 5, 9);

    T alpha, beta;

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
    real_t bReal = real_t(valueA * (float)rand() / (float)RAND_MAX);
    real_t bImag = real_t(valueB * (float)rand() / (float)RAND_MAX);

    setScalar(alpha, aReal, aImag);
    setScalar(beta, bReal, bImag);

    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    const Side side = GENERATE(Side::Left, Side::Right);

    const Op transB = GENERATE(Op::NoTrans, Op::ConjTrans, Op::Trans);

    const Diag diag = GENERATE(Diag::NonUnit, Diag::Unit);

    const Op transA = GENERATE(Op::NoTrans);

    DYNAMIC_SECTION("n = " << n << " m = " << m << " alpha = " << alpha
                           << " beta = " << beta << " Uplo = " << uplo
                           << " transB = " << transB)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(10 * n) * eps;

        idx_t nc;
        idx_t mc;
        idx_t ma;

        if (transB == Op::NoTrans) {
            nc = n;
            mc = m;
        }
        else {
            nc = m;
            mc = n;
        }

        std::vector<T> B_;
        auto B = new_matrix(B_, m, n);

        std::vector<T> C_;
        auto C = new_matrix(C_, mc, nc);

        std::vector<T> C_copy_;
        auto C_copy = new_matrix(C_copy_, mc, nc);

        if (side == Side::Right) {
            if (transB == Op::NoTrans) {
                ma = n;
            }
            else {
                ma = m;
            }
        }
        else {
            if (transB == Op::NoTrans) {
                ma = m;
            }
            else {
                ma = n;
            }
        }

        std::vector<T> A_;
        auto A = new_matrix(A_, ma, ma);

        std::vector<T> A_copy_;
        auto A_copy = new_matrix(A_copy_, ma, ma);

        MatrixMarket mm;

        mm.random(A);
        mm.random(B);
        mm.random(C);

        for (idx_t j = 0; j < ma; ++j) {
            for (idx_t i = 0; i < ma; ++i) {
                A_copy(i, j) = T(0);
            }
        }

        lacpy(Uplo::General, C, C_copy);
        lacpy(uplo, A, A_copy);

        if (diag == Diag::Unit) {
            for (idx_t i = 0; i < ma; ++i) {
                A_copy(i, i) = T(1);
            }
        }

        real_t normCbefore = lange(Norm::Fro, C_copy);
        real_t normA = lantr(Norm::Fro, uplo, diag, A);
        real_t normB = lange(Norm::Fro, B);

        trmm_out(side, uplo, transA, diag, transB, alpha, A, B, beta, C);

        if (side == Side::Right)
            gemm(transB, transA, alpha, B, A_copy, beta, C_copy);
        else {
            gemm(transA, transB, alpha, A_copy, B, beta, C_copy);
        }

        for (idx_t j = 0; j < nc; ++j) {
            for (idx_t i = 0; i < mc; ++i) {
                C(i, j) -= C_copy(i, j);
            }
        }

        real_t normC = lange(Norm::Fro, C);

        if (!(normC <= tol * ((abs(alpha) * normA * normB) +
                             (abs(beta) * normCbefore)))) {
            std::cout << "FAILING CHECK :: " << normC;
            std::cout << " <= "
                      << tol * ((abs(alpha) * normA * normB) +
                                (abs(beta) * normCbefore))
                      << std::endl;
        }

        CHECK(normC <=
              tol * ((abs(alpha) * normA * normB) + (abs(beta) * normCbefore)));
    }
}