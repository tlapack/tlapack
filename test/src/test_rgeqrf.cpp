/// @file test_rgeqrf.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/lansy.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/lapack/rgeqrf.hpp"
#include "tlapack/lapack/ung2r.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "rgeqrf utilizes geqrt3 to complete a QR factorization with a repeatedly"
    "halving block size as it moves to the right",
    "[rgeqrf]",
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

    idx_t m, n;

    m = GENERATE(5, 7, 63);
    n = GENERATE(2, 3, 5, 8, 16, 21, 51);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(100 * n) * eps;

        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> Q_;
        auto Q = new_matrix(Q_, m, n);
        std::vector<T> R_;
        auto R = new_matrix(R_, n, n);
        std::vector<T> Tmatrix_;
        auto Tmatrix = new_matrix(Tmatrix_, n, n);
        std::vector<T> tau(std::min(m, n));

        // Generate a random matrix in A
        mm.random(A);

        real_t normA, norm_orth, norm_repres;

        // Compute the norm of A
        normA = lange(FROB_NORM, A);

        // Check that the factorization was successful
        if (m <= 0 || n <= 0 || m < n) {
            norm_orth = real_t(0.0);
        }
        else {
            // Copy A to Q
            lacpy(GENERAL, A, Q);

            // 1) Compute A = QR (Stored in the matrix Q)

            // QR Factorization
            rgeqrf(Q, Tmatrix);

            // Save the R matrix
            lacpy(UPPER_TRIANGLE, Q, R);

            // Fill tau with the diagonal of the T matrix
            for (idx_t i = 0; i < n; ++i) {
                tau[i] = Tmatrix(i, i);
            }

            // Generates Q = H_1 H_2... H_n
            ung2r(Q, tau);

            // 2) Compute ||Q'Q - I||_F

            {
                std::vector<T> work_;
                auto work = new_matrix(work_, n, n);
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < n; ++i)
                        work(i, j) = static_cast<T>(0xABADBABE);

                // work receives the identity n*n
                laset(UPPER_TRIANGLE, static_cast<T>(0.0), static_cast<T>(1.0),
                      work);
                // work receives Q'Q - I
                gemm(Op::ConjTrans, Op::NoTrans, static_cast<T>(1.0), Q, Q,
                     static_cast<T>(-1.0), work);

                // Compute ||Q'Q - I||_F
                norm_orth = lansy(FROB_NORM, UPPER_TRIANGLE, work);
            }

            // 3) Compute ||QR - A||_F / ||A||_F

            {
                std::vector<T> work_;
                auto work = new_matrix(work_, m, n);
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        work(i, j) = static_cast<T>(0xABADBABE);

                // Copy Q to work
                lacpy(GENERAL, Q, work);

                trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                     static_cast<T>(1.0), R, work);

                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        work(i, j) -= A(i, j);

                norm_repres = lange(FROB_NORM, work) / normA;
            }
        }
        CHECK(norm_orth <= tol);
        CHECK(norm_repres <= tol);
    }
}
