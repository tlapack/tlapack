/// @file tik_svd.hpp Solves a Tikhonov regularized least squares problem using
/// SVD decompisition.
/// @author L. Carlos Gutierrez, Julien Langou, University of Colorado Denver,
/// USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TIK_SVD
#define TLAPACK_TIK_SVD

#include <tlapack/lapack/bidiag.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/svd_qr.hpp>
#include <tlapack/lapack/ungbr.hpp>
#include <tlapack/lapack/unmqr.hpp>

/**
 * @param[in] A is an m-by-n matrix where m >= n.
 * @param[in,out] b
 *      On entry, b is a m-by-k matrix
 *
 *      On exit, by is an m-by-k matrix that stores the solution x in the first
 *      n rows.
 * @param[in] lambda scalar
 *
 */

using namespace tlapack;

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tik_svd(matrixA_t& A, matrixb_t& b, real_t lambda)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;

    Create<matrixA_t> new_matrix;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    using range = pair<idx_t, idx_t>;

    std::vector<T> tauv(n);
    std::vector<T> tauw(n);

    // Bidiagonal decomposition
    bidiag(A, tauv, tauw);

    // Apply Q1ᴴ to b using unmqr
    //
    // Note: it is possible to use b for tmp1 and output x in b,
    // this would remove arrays btmp1 and x. Right now, we chose to have
    // the same interface for all Tikhonov functions

    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, b);

    auto x = slice(b, range{0, n}, range{0, k});

    // Extract diagonal and superdiagonal
    std::vector<real_t> d(n);
    std::vector<real_t> e(n - 1);
    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    // Allocate and initialize Q2 and P2
    std::vector<T> Q2_;
    auto Q2 = new_matrix(Q2_, n, n);
    std::vector<T> P2_;
    auto P2 = new_matrix(P2_, n, n);
    const real_t zero(0);
    const real_t one(1);
    laset(Uplo::General, zero, one, Q2);
    laset(Uplo::General, zero, one, P2);

    int err = svd_qr(Uplo::Upper, true, true, d, e, Q2, P2);

    // Apply Q2ᴴ
    std::vector<T> work_;
    auto work = new_matrix(work_, n, k);
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q2, x, work);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < k; ++i)
            work(j, i) *= (d[j] / ((d[j] * d[j]) + (lambda * lambda)));

    // Apply P2ᴴ

    gemm(CONJ_TRANS, NO_TRANS, real_t(1), P2, work, x);

    // Apply P1ᴴ

    // Finish solving least squares problem
    auto x1 = slice(x, range{1, n}, range{0, k});
    unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{0, n - 1}, range{1, n}),
          slice(tauw, range{0, n - 1}), x1);

    // Final result

    // lacpy(GENERAL, x4, x);
}

#endif  // TLAPACK_TIK_SVD