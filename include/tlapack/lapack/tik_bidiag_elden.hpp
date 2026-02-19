/// @file tik_bidiag_elden.hpp  Solves a Tikhonov regularized least squares
/// problem using Eldén's bidiagonalization algorithm.
/// @author L. Carlos Gutierrez, Julien Langou, University of Colorado Denver,
/// USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TIK_BIDIAG_ELDEN_HH
#define TLAPACK_TIK_BIDIAG_ELDEN_HH

#include <tlapack/lapack/bidiag.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ungbr.hpp>
#include <tlapack/lapack/unmlq.hpp>
#include <tlapack/lapack/unmqr.hpp>

#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"
#include "tlapack/lapack/elden_elim.hpp"

/**
 * @brief This function solves the standard Tikhonov regularized least squares
 * problem using Eldén's bidiagonalization algorithm.
 *
 * See: L. Eldén. Algorithms for the regularization of ill-conditioned least
 * square problems. BIT, 17:134–145, 1977
 *
 * @param[in,out] A is an m-by-n matrix where m >= n.
 *      On exit, A is trashed.
 * @param[in,out] b
 *      On entry, b is a m-by-k matrix
 *      On exit, b stores the solution x in the first n rows.
 * @param[in] lambda scalar
 *      The famous Tikhonov regularization parameter
 *
 */

using namespace tlapack;

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tik_bidiag_elden(matrixA_t& A, matrixb_t& b, real_t lambda)
{
    // check arguments
    tlapack_check(nrows(A) >= ncols(A));
    tlapack_check(nrows(b) == nrows(A));

    // Initliazation for basic utilities
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;

    using range = pair<idx_t, idx_t>;

    Create<matrixA_t> new_matrix;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    std::vector<T> work_;
    auto work = new_matrix(work_, n, k);

    std::vector<T> tauv(n);
    std::vector<T> tauw(n);
    std::vector<real_t> d(n);
    std::vector<real_t> e(n - 1);

    // x is a view of b
    auto x_view_b = slice(b, range{0, n}, range{0, k});

    bidiag(A, tauv, tauw);

    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, b);

    // Extract diagonal diagonal d and super diagonal e from decomposed A
    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    elden_elim(lambda, d, e, x_view_b, work);

    // Solve for x without constructing P1 using views
    auto view_b = slice(b, range{1, n}, range{0, k});

    // Note: x stored in the upper part of b
    unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{0, n - 1}, range{1, n}),
          slice(tauw, range{0, n - 1}), view_b);
}

#endif  // TLAPACK_TIK_BIDIAG_ELDEN_HH