/// @file test_hemm2.cpp
/// @author Brian Dang, University of Colorado Denver, USA
/// @brief Test LLH multiplication
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "TestUploMatrix.hpp"

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/axpy.hpp>
#include <tlapack/blas/hemm.hpp>
#include <tlapack/blas/hemm2.hpp>

using namespace tlapack;

#define TESTUPLO_TYPES_TO_TEST                                          \
    (TestUploMatrix<float, size_t, Uplo::Lower, Layout::ColMajor>),     \
        (TestUploMatrix<float, size_t, Uplo::Upper, Layout::ColMajor>), \
        (TestUploMatrix<float, size_t, Uplo::Lower, Layout::RowMajor>), \
        (TestUploMatrix<float, size_t, Uplo::Upper, Layout::RowMajor>)

/// Print matrix A in the standard output
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
}

TEMPLATE_TEST_CASE("mult a triangular matrix with a rectangular matrix",
                   "[hemm2]",
                   TLAPACK_TYPES_TO_TEST,
                   TESTUPLO_TYPES_TO_TEST)

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

    const idx_t m = GENERATE(3, 8, 6, 8);
    const idx_t n = GENERATE(5, 6, 4, 15);

    const Side side = GENERATE(Side::Left, Side::Right);
    const Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);
    const Op transB = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);

    T alpha;
    T beta;

    if constexpr (is_complex<T>) {
        auto a_real = GENERATE(1, 2, -7, 8.6);
        auto a_imag = GENERATE(1, 0, -7, 8.6);
        auto b_real = GENERATE(1, 2, -4, 6.5);
        auto b_imag = GENERATE(1, 0, -4, 6.5);

        alpha = T(a_real, a_imag);
        beta = T(b_real, b_imag);
    }
    else {
        alpha = GENERATE(1, 2, -7, 8.6);
        beta = GENERATE(1, 2, -4, 6.5);
    }

    bool verbose = false;

    DYNAMIC_SECTION("n = " << n << " m = " << m << " side = " << side
                           << " uplo = " << uplo << " op = " << transB
                           << " alpha = " << alpha << " beta = " << beta)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        // Create Matrices
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);

        std::vector<T> B_;
        auto B =
            (side == Side::Left) ? new_matrix(B_, m, n) : new_matrix(B_, n, m);

        std::vector<T> BT_;
        auto BT = (side == Side::Left) ? new_matrix(BT_, n, m)
                                       : new_matrix(BT_, m, n);

        std::vector<T> ansHemm_;
        auto ansHemm = (side == Side::Left) ? new_matrix(ansHemm_, n, m)
                                            : new_matrix(ansHemm_, m, n);

        std::vector<T> ansHemm2_;
        auto ansHemm2 = (side == Side::Left) ? new_matrix(ansHemm2_, n, m)
                                             : new_matrix(ansHemm2_, m, n);

        // Update A with random numbers, and make it positive definite
        mm.random(uplo, A);
        for (idx_t j = 0; j < n; ++j) {
            if constexpr (is_complex<T>) {
                A(j, j) = T(real(A(j, j)) + n, 0);
            }
            else {
                A(j, j) = A(j, j) + n;
            }
        }
        if (verbose) {
            std::cout << "\nA = ";
            printMatrix(A);
        }

        // Fill in B with random numbers
        mm.random(B);
        if (verbose) {
            std::cout << "\nB = ";
            printMatrix(B);
        }

        // Create the B transpose
        if (side == Side::Left) {
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < m; j++) {
                    if (transB == Op::ConjTrans) {
                        BT(i, j) = conj(B(j, i));
                    }
                    else {
                        BT(i, j) = B(j, i);
                    }
                }
            }
        }
        else {
            for (idx_t i = 0; i < m; i++) {
                for (idx_t j = 0; j < n; j++) {
                    if (transB == Op::ConjTrans) {
                        BT(i, j) = conj(B(j, i));
                    }
                    else {
                        BT(i, j) = B(j, i);
                    }
                }
            }
        }

        if (verbose) {
            std::cout << "\nBT = ";
            printMatrix(BT);
        }

        mm.random(ansHemm);
        if (verbose) {
            std::cout << "\nansHemm = ";
            printMatrix(ansHemm);
        }

        lacpy(GENERAL, ansHemm, ansHemm2);
        if (verbose) {
            std::cout << "\nansHemm2 = ";
            printMatrix(ansHemm2);
        }

        // Fill in zeroes
        if (uplo == Uplo::Lower) {
            auto subMatrix = slice(A, range(0, n - 1), range(1, n));
            laset(UPPER_TRIANGLE, real_t(0), real_t(0), subMatrix);
        }
        else {
            auto subMatrix = slice(A, range(1, n), range(0, n - 1));
            laset(LOWER_TRIANGLE, real_t(0), real_t(0), subMatrix);
        }
        if (verbose) {
            std::cout << "\nAfter Slice A = ";
            printMatrix(A);
        }

        // Do Hemm
        hemm(side, uplo, alpha, A, BT, beta, ansHemm);
        real_t normHemm = lange(FROB_NORM, ansHemm);
        if (verbose) {
            std::cout << "\nthis is ansHemm";
            printMatrix(ansHemm);
            std::cout << std::endl;
        }

        // Do Hemm2 If No Trans use BT
        (transB == Op::NoTrans)
            ? hemm2(side, uplo, transB, alpha, A, BT, beta, ansHemm2)
            : hemm2(side, uplo, transB, alpha, A, B, beta, ansHemm2);
        if (verbose) {
            std::cout << "\nthis is ansHemm2";
            printMatrix(ansHemm2);
            std::cout << std::endl;
        }

        // ansHemm2 -= ansHemm
        if (side == Side::Left) {
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < m; j++) {
                    ansHemm2(i, j) -= ansHemm(i, j);
                }
            }
        }
        else {
            for (idx_t i = 0; i < m; i++) {
                for (idx_t j = 0; j < n; j++) {
                    ansHemm2(i, j) -= ansHemm(i, j);
                }
            }
        }

        if (verbose) {
            std::cout << "\nThis is the final answer";
            printMatrix(ansHemm2);
            std::cout << std::endl;
        }

        // Check for relative error: norm(A-cholesky(A))/norm(A)
        real_t error = lange(FROB_NORM, ansHemm2) / normHemm;
        CHECK(error <= tol);
    }
}
