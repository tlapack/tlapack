/// @file test_hemm2.cpp
/// @author Brian Dang, University of Colorado Denver, USA
/// @brief Test hermition matrix multiplication
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)

#include "TestUploMatrix.hpp"

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/hemm.hpp>
#include <tlapack/lapack/hemm2.hpp>

using namespace tlapack;

#define TESTUPLO_TYPES_TO_TEST                                             \
    (TestUploMatrix<float, size_t, LOWER_TRIANGLE, Layout::ColMajor>),     \
        (TestUploMatrix<float, size_t, UPPER_TRIANGLE, Layout::ColMajor>), \
        (TestUploMatrix<float, size_t, LOWER_TRIANGLE, Layout::RowMajor>), \
        (TestUploMatrix<float, size_t, UPPER_TRIANGLE, Layout::RowMajor>)

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

    const idx_t m = GENERATE(3, 6);
    const idx_t n = GENERATE(2, 5);

    const Side side = GENERATE(Side::Left, Side::Right);
    const Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);
    const Op transB = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);

    T alpha, beta;

    srand(3);

    real_t aReal = (float)rand();
    real_t aImag = (float)rand();
    real_t bReal = (float)rand();
    real_t bImag = (float)rand();

    setScalar(alpha, aReal, aImag);
    setScalar(beta, bReal, bImag);

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
            (side == LEFT_SIDE) ? new_matrix(B_, m, n) : new_matrix(B_, n, m);

        std::vector<T> BT_;
        auto BT =
            (side == LEFT_SIDE) ? new_matrix(BT_, n, m) : new_matrix(BT_, m, n);

        std::vector<T> C_;
        auto C =
            (side == LEFT_SIDE) ? new_matrix(C_, n, m) : new_matrix(C_, m, n);

        std::vector<T> D_;
        auto D =
            (side == LEFT_SIDE) ? new_matrix(D_, n, m) : new_matrix(D_, m, n);

        // Update A with random numbers, and make it positive definite
        mm.random(uplo, A);
        for (idx_t j = 0; j < n; ++j) {
            A(j, j) = real_t(n + j);
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
        if (side == LEFT_SIDE) {
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < m; j++) {
                    if (transB == CONJ_TRANS) {
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
                    if (transB == CONJ_TRANS) {
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

        // Random C value
        mm.random(C);
        if (verbose) {
            std::cout << "\nC = ";
            printMatrix(C);
        }

        // Copy the C value
        lacpy(GENERAL, C, D);
        if (verbose) {
            std::cout << "\nD = ";
            printMatrix(D);
        }

        // Fill in zeroes
        if (uplo == LOWER_TRIANGLE) {
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
        hemm(side, uplo, alpha, A, BT, beta, C);
        real_t normHemm = lange(FROB_NORM, C);
        if (verbose) {
            std::cout << "\nthis is C";
            printMatrix(C);
            std::cout << std::endl;
        }

        // Do Hemm2 If No Trans use BT
        (transB == NO_TRANS) ? hemm2(side, uplo, transB, alpha, A, BT, beta, D)
                             : hemm2(side, uplo, transB, alpha, A, B, beta, D);
        if (verbose) {
            std::cout << "\nthis is D";
            printMatrix(D);
            std::cout << std::endl;
        }

        // D -= C
        if (side == LEFT_SIDE) {
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < m; j++) {
                    D(i, j) -= C(i, j);
                }
            }
        }
        else {
            for (idx_t i = 0; i < m; i++) {
                for (idx_t j = 0; j < n; j++) {
                    D(i, j) -= C(i, j);
                }
            }
        }

        if (verbose) {
            std::cout << "\nThis is the final answer";
            printMatrix(D);
            std::cout << std::endl;
        }

        // Check for relative error: norm(A-cholesky(A))/norm(A)
        real_t error = lange(FROB_NORM, D) / normHemm;
        CHECK(error <= tol);
    }
}