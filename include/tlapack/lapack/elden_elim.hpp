/// @file tik_elim.hpp
/// @brief Solves a Tikhonov regularized least squares problem using Eldén's
/// bidiagonalization algorithm.
/// @author L. Carlos Gutierrez, University of Colorado Denver, USA
/// @author Julien Langou, University of Colorado Denver, USA
//
/// @copyright
/// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ELDEN_ELIM_HPP
#define TLAPACK_ELDEN_ELIM_HPP

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

/**
 * @param[in]      lambda  Scalar parameter for the elimination.
 * @param[in,out]  d       Vector of length n. On entry, contains the diagonal
 *                         of the bidiagonal matrix; on exit, contains the
 *                         updated diagonal.
 * @param[in,out]  e       Vector of length n–1. On entry, contains the
 *                         superdiagonal; on exit, contains the updated
 *                         superdiagonal.
 * @param[in,out]  b       Matrix of size n × k. On entry, contains the
 *                         projected vector; on exit, stores the updated
 *                         vector b after elimination.
 * @param[in,out]  work    Workspace matrix (n × k). On entry, may be
 * uninitialized; on exit, used to compute residuals of the full augmented
 * vector b.
 */

using namespace tlapack;

template <TLAPACK_REAL real_t,
          TLAPACK_VECTOR vectord_t,
          TLAPACK_VECTOR vectore_t,
          TLAPACK_WORKSPACE work_t,
          TLAPACK_SMATRIX matrix_t>
void elden_elim(
    real_t lambda, vectord_t& d, vectore_t& e, matrix_t& b, work_t& work)
{
    tlapack_check(size(d) == size(e) + 1);
    tlapack_check(size(d) == nrows(b));
    tlapack_check((nrows(b) == nrows(work)) && (ncols(b) == ncols(work)));

    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    idx_t n = size(d);
    idx_t k = ncols(b);

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
                work(0, j) = -sn * b(0, j);
                b(0, j) = cs * b(0, j);
            }
        }
        else {
            auto bv = slice(b, i, range{0, k});
            auto cv = slice(work, i, range{0, k});
            rot(bv, cv, cs, sn);
        }

        ////////// Elimate byproducts /////////////

        if (i < n - 1) {
            // Step a: generate rotation

            low_d = lambda;
            rotg(low_d, low_e, cs, sn);

            // Step c: update right hand side b

            for (idx_t j = 0; j < k; ++j) {
                work(i + 1, j) = sn * work(i, j);
                work(i, j) = cs * work(i, j);
            }
        }
    }

    auto xS0 = slice(b, n - 1, range{0, k});
    rscl(d[n - 1], xS0);
    for (idx_t i = n - 1; i-- > 0;) {
        auto xS0 = slice(b, i, range{0, k});
        auto xS1 = slice(b, i + 1, range{0, k});
        axpy(-e[i], xS1, xS0);
        rscl(d[i], xS0);
    }
}
#endif