/// @file test_potri.cpp
/// @author L. Carlos Gutierrez, Brian Dang, and Henricus Bouwmeester,
/// University of Colorado Denver, USA
/// @brief Test Potri Function
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "TestUploMatrix.hpp"

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/lapack/mult_hehe.hpp>
#include <tlapack/lapack/potri.hpp>

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

TEMPLATE_TEST_CASE("compute the inverse of a hermitian matrix",
                   "[potri]",
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

    const idx_t n = GENERATE(5, 6, 10, 12);

    const Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);

    bool verbose = false;

    DYNAMIC_SECTION("n = " << n << " uplo = " << uplo)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        // Create Matrices
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);

        std::vector<T> B_;
        auto B = new_matrix(B_, n, n);

        std::vector<T> C_;
        auto C = new_matrix(C_, n, n);

        std::vector<T> I_;
        auto I = new_matrix(I_, n, n);

        // Update A with random numbers, and make it positive definite
        mm.random(uplo, A);
        for (idx_t j = 0; j < n; ++j) {
            A(j, j) = real_t(n + j);
        }
        if (verbose) {
            std::cout << "\nA = ";
            printMatrix(A);
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
        real_t normA = lange(FROB_NORM, A);

        // Zero out the entire matrix first
        laset(GENERAL, real_t(0), real_t(0), I);
        // Set diagonal to 1
        for (idx_t i = 0; i < n; ++i)
            I(i, i) = real_t(1);
        if (verbose) {
            std::cout << "\nI = ";
            printMatrix(I);
        }

        // Copy A into B
        lacpy(GENERAL, A, B);
        if (verbose) {
            std::cout << "\nB should be a copy of A = ";
            printMatrix(B);
        }

        potri(uplo, B);
        if (verbose) {
            std::cout << "\nB is inversed = ";
            printMatrix(B);
        }
        real_t normAIn = lange(FROB_NORM, B);

        mult_hehe(uplo, real_t(1), A, B, real_t(0), C);
        if (verbose) {
            std::cout << "\nThis should look like Identity = ";
            printMatrix(C);
        }

        // C - A
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = 0; i < n; i++) {
                C(i, j) -= I(i, j);
            }
        }

        real_t error = lange(FROB_NORM, C) / normA / normAIn;
        CHECK(error <= tol);
    }
}