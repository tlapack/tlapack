/// @file test_gelqt3.cpp
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
#include "tlapack/lapack/gelqt3.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/laset.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "gelqt3 recursively computes the LQ factorization of a matrix",
    "[gelqt3]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    MatrixMarket mm;

    idx_t m, n;
    m = GENERATE(1, 2, 3, 5, 8, 15, 16, 17, 31, 32, 63, 64);
    n = GENERATE(1, 2, 4, 8, 16, 33, 64, 65, 128);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100 * n) * eps;

    // Matrices
    // original matrix A
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    // Lower triangular
    std::vector<T> L_;
    auto L = new_matrix(L_, m, m);
    // Householder vectors
    std::vector<T> V_;
    auto V = new_matrix(V_, m, n);
    // A copy
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<T> T_;
    auto Tmatrix = new_matrix(T_, m, m);
    // Used to compute ||Qᴴ Q - I||ꜰ
    std::vector<T> work_;
    auto work = new_matrix(work_, m, m);

    real_t norm_orth = real_t(0.0);
    real_t normA = real_t(0.0);
    real_t norm_repres = real_t(0.0);

    if (m > 0 && n > 0 && m <= n) {
        // Generate a random matrix in A & T
        mm.random(A);

        normA = lange(FROB_NORM, A);
        // Copy A to Q
        lacpy(GENERAL, A, Q);

        laset(GENERAL, T(0.0), T(0.0), Tmatrix);
        // 1) Compute the LQ factorization of A
        gelqt3(Q, Tmatrix);

        lacpy(Uplo::Lower, Q, L);

        // Copy the Householder vectors into V
        lacpy(GENERAL, Q, V);
        // creates the identity matrix in Q
        laset(GENERAL, static_cast<T>(0.0), static_cast<T>(1.0), Q);
        // Apply the Householder reflectors
        larfb(Side::Right, Op::ConjTrans, Direction::Forward, StoreV::Rowwise,
              V, Tmatrix, Q);
        // work receives the identity n*n
        laset(GENERAL, static_cast<T>(0.0), static_cast<T>(1.0), work);
        // work receives Qᴴ Q - I
        gemm(Op::NoTrans, Op::ConjTrans, static_cast<T>(1.0), Q, Q,
             static_cast<T>(-1.0), work);

        norm_orth = lange(FROB_NORM, work);

        // 3) Compute ||LQ - A||ꜰ / ||A||ꜰ

        trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit,
             static_cast<T>(1.0), L, Q);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                Q(i, j) -= A(i, j);

        norm_repres = lange(FROB_NORM, Q) / normA;
    }
    CHECK(norm_repres <= tol);
    CHECK(norm_orth <= tol);
}