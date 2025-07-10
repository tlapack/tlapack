/// @file elden_elim.hpp Uses givens rotations to eliminate lambda from the
/// bidiagonal L matrix utilized in Eldén's bidiagonalization algorithm. .
/// @author L. Carlos Gutierrez, Julien Langou, University of Colorado Denver,
/// USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
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
 * @param[in] lambda scalar.
 * @param[in, out] d
 *      -On entry, d is vector of size(n) that contains the diagonal of a
 *      bidiagonal matrix.
 *
 *      -On exit, d contains the updated diagonal of the elimination algorithm.
 * @param[in, out] e
 *      -On entry, e is vector of size(n - 1) that contains the super diagonal
 *      of a bidiagonal matrix.
 *
 *      -On exit, e is the updated super diagonal of the elimination algorithm.
 * @param[in, out] work
 *      -On entry, work is an n-by-k matrix that is a workspace filled with
 *      junk.
 *
 *      -On exit, work is used to compute the residual of the full augmented
 *      vector b
 * @param[in, out] b
 *      -On entry, b is a matrix of size n-by-k which stores the projected
 *      vector .b
 *
 *      -On exit, b is a matrix that stores the updated vector b after the
 *      elimination algorithm.
 */

using namespace tlapack;

template <TLAPACK_REAL real_t,
          TLAPACK_VECTOR vectord_t,
          TLAPACK_VECTOR vectore_t,
          TLAPACK_WORKSPACE work_t,
          TLAPACK_SMATRIX matrix_t>
void elden_elim(
    real_t lambda, vectord_t& d, vectore_t& e, work_t& work, matrix_t& b)
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