/// @file tik_bidiag_elden.hpp  Solves a tikhonov regularized least squares
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
#include <tlapack/lapack/unmqr.hpp>

#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

/**
 * @brief This function solves the least squares problem for a Tikhonov
 * regularized matrix using Eldén's bidiagonalization algorithm.
 *
 * See: L. Eldén. Algorithms for the regularization of ill-conditioned least
 * square problems. BIT, 17:134–145, 1977
 *
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

// /// Solves tikhonov regularized least squares using special bidiag method
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tik_bidiag_elden(matrixA_t& A, matrixb_t& b, real_t lambda)
{
    // check arguments
    tlapack_check(nrows(A) >= ncols(A));

    // Initliazation for basic utilities
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;

    using range = pair<idx_t, idx_t>;

    Create<matrixA_t> new_matrix;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    // Bidiagonal decomposition
    std::vector<T> tauv(n);
    std::vector<T> tauw(n);

    bidiag(A, tauv, tauw);
    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, b);

    // Initializiations for bidiag specialized decomposition

    // Extract diagonal diagonal d and super diagonal e from decomposed A
    std::vector<real_t> d(n);
    std::vector<real_t> e(n - 1);

    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    // Declare and initialize baug
    std::vector<T> baug_;
    auto baug = new_matrix(baug_, 2 * n, k);

    // Augment zeros onto b
    auto baug_top = slice(baug, range{0, n}, range{0, k});
    auto baug_bottom = slice(baug, range{n, n + n}, range{0, k});
    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), baug_top);
    laset(GENERAL, real_t(0), real_t(0), baug_bottom);

    //////////// Algorithm for eliminating lambda*I /////////////

    real_t low_d = lambda;
    real_t low_e;
    real_t cs, sn;

    for (idx_t i = 0; i < n; i++) {
        ///////// Eliminate lambdas on diagonal /////////

        // Step a: generate rotation

        rotg(d[i], low_d, cs, sn);

        // Step b: apply to next column in A
        if (i < n - 1) {
            low_e = -sn * e[i];
            e[i] = cs * e[i];
        }
        // Step c: update right hand side b

        if (i == 0) {
            for (idx_t j = 0; j < k; ++j) {
                baug(n, j) = -sn * baug(0, j);
                baug(0, j) = cs * baug(0, j);
            }
        }
        else {
            auto bv = slice(baug, i, range{0, k});
            auto cv = slice(baug, n + i, range{0, k});
            rot(bv, cv, cs, sn);
        }

        ////////// Elimate byproducts /////////////

        if (i < n - 1) {
            // Step a: generate rotation

            low_d = lambda;
            rotg(low_d, low_e, cs, sn);

            // Step c: update right hand side b

            for (idx_t j = 0; j < k; ++j) {
                baug(n + 1 + i, j) = sn * baug(n + i, j);
                baug(n + i, j) = cs * baug(n + i, j);
            }
        }
    }

    // Solve the least squares problem

    // Bidiag solving algorithm
    auto xS = slice(baug, range{0, n}, range{0, k});
    auto xS0 = slice(baug, n - 1, range{0, k});
    rscl(d[n - 1], xS0);
    for (idx_t i = n - 1; i-- > 0;) {
        auto xS0 = slice(baug, i, range{0, k});
        auto xS1 = slice(baug, i + 1, range{0, k});
        axpy(-e[i], xS1, xS0);
        rscl(d[i], xS0);
    }

    // Finish solving least sqaures problem
    std::vector<T> P1_;
    auto P1 = new_matrix(P1_, n, n);
    lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);

    ungbr_p(n, P1, tauw);
    auto x = slice(b, range{0, n}, range{0, k});
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, xS, x);

    // auto x2 = slice(b, range{1, n}, range{0, k});
    // unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{1, n}, range{1, n}),
    //       slice(tauw, range{0, n - 1}), x2);
}

#endif  // TLAPACK_TIK_BIDIAG_ELDEN_HH