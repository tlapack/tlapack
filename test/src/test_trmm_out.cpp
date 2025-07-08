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
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lantr.hpp>
#include <tlapack/lapack/mult_hehe.hpp>
#include <tlapack/lapack/trmm_out.hpp>

using namespace tlapack;

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
    std::cout << std::endl;
}

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
    // const Uplo uplo = GENERATE(Uplo::Upper);
    // const Op transB = GENERATE(Op::NoTrans);

    const Side side = GENERATE(Side::Right);

    const Op transB = GENERATE(Op::NoTrans, Op::ConjTrans, Op::Trans);

    // const Diag diag = GENERATE(Diag::NonUnit);

    const Diag diag = GENERATE(Diag::NonUnit, Diag::Unit);

    DYNAMIC_SECTION("n = " << n << " m = " << m << " alpha = " << alpha
                           << " beta = " << beta << " Uplo = " << uplo
                           << " transB = " << transB)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        idx_t nc;
        idx_t mc;
        idx_t ma;

        if (transB == Op::NoTrans) {
            nc = n;
            mc = m;
            std::cout << "nc = n = " << nc << " and mc = m = " << mc
                      << std::endl;
        }
        else {
            nc = m;
            mc = n;
            std::cout << "nc = m = " << nc << " and mc = n = " << mc
                      << std::endl;
        }

        std::vector<T> B_(m * n);
        tlapack::LegacyMatrix<T> B(m, n, &B_[0], m);

        std::vector<T> C_(mc * nc);
        tlapack::LegacyMatrix<T> C(mc, nc, &C_[0], mc);

        std::vector<T> C_copy_(mc * nc);
        tlapack::LegacyMatrix<T> C_copy(mc, nc, &C_copy_[0], mc);

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

        std::vector<T> A_(ma * ma);
        tlapack::LegacyMatrix<T> A(ma, ma, &A_[0], ma);

        std::vector<T> A_copy_(ma * ma);
        tlapack::LegacyMatrix<T> A_copy(ma, ma, &A_copy_[0], ma);

        std::cout << "initialized matrices" << std::endl;

        MatrixMarket mm;

        mm.random(A);
        mm.random(B);
        mm.random(C);

        // std::cout << "B = " << std::endl;
        // printMatrix(B);
        // std::cout << "A = " << std::endl;
        // printMatrix(A);
        // std::cout << "C = " << std::endl;
        // printMatrix(C);

        for (idx_t j = 0; j < ma; ++j) {
            for (idx_t i = 0; i < ma; ++i) {
                A_copy(i, j) = 0;
            }
        }

        std::cout << "Acopy to 0" << std::endl;
        lacpy(Uplo::General, C, C_copy);
        lacpy(uplo, A, A_copy);

        if (diag == Diag::Unit) {
            for (idx_t i = 0; i < ma; ++i) {
                A_copy(i, i) = 1;
            }
        }

        real_t normCbefore = lange(Norm::Fro, C_copy);
        real_t normA = lantr(Norm::Fro, uplo, diag, A);
        real_t normB = lange(Norm::Fro, B);
        std::cout << "norms before" << std::endl;
        
        std::cout << "nrows and cols of A = " << nrows(A_copy) << ", " << ncols(A_copy) << " nrows/cols of B = " << nrows(B) << ", " << ncols(B) << " nrows/cols of C = " << nrows(C_copy) << ", " << ncols(C_copy) << std::endl;

        trmm_out(side, uplo, Op::NoTrans, diag, transB, alpha,
                 A, B, beta, C);
        std::cout << "done trmm" << std::endl;

        if (side == Side::Right)
            gemm(transB, Op::NoTrans, alpha, B, A_copy, beta, C_copy);
        else {
            std::cout << "nrows and cols of A = " << nrows(A_copy) << ", " << ncols(A_copy) << " nrows/cols of B = " << nrows(B) << ", " << ncols(B) << " nrows/cols of C = " << nrows(C_copy) << ", " << ncols(C_copy) << std::endl;
            gemm(Op::NoTrans, transB, alpha, A_copy, B, beta, C_copy);
        }

        std::cout << "done gemm" << std::endl;

        for (idx_t j = 0; j < nc; ++j) {
            for (idx_t i = 0; i < mc; ++i) {
                C(i, j) -= C_copy(i, j);
            }
        }

        std::cout << "done subtraction" << std::endl;
        printMatrix(C);

        real_t normC = lange(Norm::Fro, C);

        normC = normC /
                ((real(alpha) * normA * normB) + (real(beta) * normCbefore));

        CHECK(normC <= tol);
    }
}