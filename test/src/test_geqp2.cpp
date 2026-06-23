/// @file test_geqp2.cpp
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
#include <chrono>  // for high_resolution_clock
#include <random>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/swap.hpp>
#include <tlapack/lapack/geqp2.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/unmqr.hpp>

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    // constants
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
    std::cout << std::endl;
}

using namespace tlapack;

TEMPLATE_TEST_CASE("geqp2 computes the QR factorization of a matrix",
                   "[geqp2]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    MatrixMarket mm;

    idx_t m, n, r;

    /*
    r = GENERATE(1, 2, 3, 5, 15, 20);
    // Generate n to be strictly greater than r
    n = GENERATE_COPY(r + 1, r + 2, r + 5, r + 10);
    // Generate m to be strictly greater than n
    m = GENERATE_COPY(n + 1, n + 2, n + 15);
    */

    r = GENERATE(3);
    n = GENERATE_COPY(r + 2);
    m = GENERATE_COPY(n + 2);

    std::cout << std::endl
              << "(m,n,r) = (" << m << ", " << n << ", " << r << ")"
              << std::endl;

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100 * n) * eps;

    // Matrices
    // original matrix A
    // Arrays
    std::vector<idx_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<T> tau(n);
    std::vector<real_t> vn1(n);
    std::vector<real_t> vn2(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> S_;
    auto S = new_matrix(S_, m, r);
    std::vector<T> C_;
    auto C = new_matrix(C_, r, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);

    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);

    // Initialize the Rank
    idx_t rank = 0;

    // real_t normA = real_t(0.0);
    real_t recon_error = real_t(0.0);
    real_t orthogonality_error = real_t(0.0);

    if (m > 0 && n > 0 && m >= n) {
        // Generate a random matrix in A
        mm.random(S);
        mm.random(C);

        //------------------------------------------------------------------

        // 1) Compute A = S * C

        // Perform Matrix multiplication A = 1.0*S*C + 0.0*A
        gemm(Op::NoTrans, Op::NoTrans, T(1.0), S, C, T(0.0), A);

        std::cout << std::endl << "A = " << std::endl;
        printMatrix(A);
        

        // Copy A to A_orig & elements of tau_head and tau_tail into tau
        lacpy(GENERAL, A, A_orig);

        geqp2(A, tau, perm, vn1, vn2);
        std::cout << std::endl << "A = " << std::endl;
        printMatrix(A);

        // printMatrix(A);

        std::cout << std::flush;

        // 7) Evaluate Numerical Rank
        // Set absolute threshold: tolerance = max(m,n) * epsilon * |R(0,
        // 0)|
        real_t tol = std::max(m, n) * eps * std::abs(A(0, 0));
        for (idx_t i = 0; i < std::min(m, n); i++) {
            if (std::abs(A(i, i)) > tol) {
                rank++;
            }
            else {
                break;
            }
        }

        laset(GENERAL, T(0.0), T(0.0), R);
        lacpy(UPPER_TRIANGLE, A, R);

        laset(GENERAL, T(0.0), T(0.0), Q);
        lacpy(GENERAL, A, Q);

        ung2r(Q, tau);
        std::cout << std::endl << "Q = " << std::endl;
        printMatrix(Q);

        // Compute the reconstruction accuracy ||A - Q*R||_F.
        gemm(Op::NoTrans, Op::NoTrans, T(-1.0), Q, R, T(1.0), A);
        recon_error = lange(FROB_NORM, A) / lange(FROB_NORM, A_orig);

        // Compute the orthogonality error ||I - Q^H Q||_F.
        laset(GENERAL, T(0.0), T(1.0), work);
        gemm(Op::ConjTrans, Op::NoTrans, T(-1.0), Q, Q, T(1.0), work);
        orthogonality_error = lange(FROB_NORM, work);

        //---------------------------------------------------
    }
    std::cout << std::endl
              << "||A - Q*R||_F / ||A||_F = " << recon_error << std::endl;
    std::cout << "||I - Q^H Q||_F = "
              << orthogonality_error / static_cast<real_t>(n) << std::endl;

    CHECK(recon_error <= tol);
    CHECK(orthogonality_error <= tol);
    CHECK(r == rank);
}
