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
    using tlapack::swap;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    MatrixMarket mm;

    idx_t m, n, r;

    r = GENERATE(3, 5, 7);
    // Generate n to be greater than or equal to r
    n = GENERATE(7, 11, 15);
    // Generate m to be greater than or equal to n
    m = GENERATE(15, 21, 30);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100 * n) * eps;

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

    real_t normA = real_t(0.0);
    real_t recon_error = real_t(0.0);
    real_t orthogonality_error = real_t(0.0);

    if (m > 0 && n > 0 && m >= n) {
        // Generate a random matrix in A
        mm.random(S);
        mm.random(C);

        // 1) Compute A = S * C

        // Perform Matrix multiplication A = 1.0*S*C + 0.0*A
        gemm(Op::NoTrans, Op::NoTrans, T(1.0), S, C, T(0.0), A);

        // Copy A to A_orig and compute the norm
        lacpy(GENERAL, A, A_orig);
        normA = lange(FROB_NORM, A_orig);

        // 2) Compute QR factorization using Drmac's algorithm
        geqp2(A, tau, perm, vn1, vn2);

        // 3) Permute the original matrix according to what GEQP2 provides
        std::vector<bool> visited(n, false);
        for (idx_t i = 0; i < n; ++i) {
            if (visited[i] || perm[i] == i) {
                continue;
            }

            idx_t current = i;
            while (!visited[current]) {
                visited[current] = true;
                idx_t next_node = perm[current];
                if (!visited[next_node]) {
                    auto current_col = col(A_orig, current);
                    auto next_col = col(A_orig, next_node);

                    tlapack::swap(current_col, next_col);

                    current = next_node;
                }
            }
        }

        // 4) Evaluate Numerical Rank
        real_t tol_rank = real_t(max(m, n) * eps * normA);
        for (idx_t i = 0; i < min(m, n); i++) {
            if (abs(A(i, i)) > tol_rank) {
                rank++;
            }
        }

        // 5) Save R
        laset(GENERAL, T(0.0), T(0.0), R);
        lacpy(UPPER_TRIANGLE, A, R);

        // 6) Create Q
        laset(GENERAL, T(0.0), T(0.0), Q);
        lacpy(GENERAL, A, Q);
        ung2r(Q, tau);

        // 7) Compute the reconstruction accuracy ||A - Q*R||_F.
        gemm(Op::NoTrans, Op::NoTrans, T(-1.0), Q, R, T(1.0), A_orig);
        recon_error = lange(FROB_NORM, A_orig) / normA;

        // 8) Compute the orthogonality error ||I - Q^H Q||_F.
        laset(GENERAL, T(0.0), T(1.0), work);
        gemm(Op::ConjTrans, Op::NoTrans, T(-1.0), Q, Q, T(1.0), work);
        orthogonality_error = lange(FROB_NORM, work);
    }

    CHECK(recon_error <= tol);
    CHECK(orthogonality_error <= tol);
    CHECK(r == rank);
}
