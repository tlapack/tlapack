/// @file generalized_schur_swap.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaexc.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GENERALIZED_SCHUR_SWAP_HH
#define TLAPACK_GENERALIZED_SCHUR_SWAP_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/lapack/lahqr_schur22.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/lasy2.hpp"

namespace tlapack {

/** schur_swap, swaps 2 eigenvalues of A.
 *
 * @return  0 if success
 * @return  1 the swap failed, this usually means the eigenvalues
 *            of the blocks are too close.
 *
 * @param[in]     want_q bool
 *                Whether or not to apply the transformations to Q
 * @param[in]     want_z bool
 *                Whether or not to apply the transformations to Z
 * @param[in,out] A n-by-n matrix.
 *                Must be in Schur form
 * @param[in,out] B n-by-n matrix.
 *                Must be in Schur form
 * @param[in,out] Q n-by-n matrix.
 *                Orthogonal matrix, not referenced if want_q is false
 * @param[in,out] Z n-by-n matrix.
 *                Orthogonal matrix, not referenced if want_q is false
 * @param[in]     j0 integer
 *                Index of first eigenvalue block
 * @param[in]     n1 integer
 *                Size of first eigenvalue block
 * @param[in]     n2 integer
 *                Size of second eigenvalue block
 *
 * @ingroup auxiliary
 */
template <TLAPACK_CSMATRIX matrix_t,
          enable_if_t<is_real<type_t<matrix_t>>, bool> = true>
int generalized_schur_swap(bool want_q,
                           bool want_z,
                           matrix_t& A,
                           matrix_t& B,
                           matrix_t& Q,
                           matrix_t& Z,
                           const size_type<matrix_t>& j0,
                           const size_type<matrix_t>& n1,
                           const size_type<matrix_t>& n2)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Functor for creating new matrices
    Create<matrix_t> new_matrix;

    const idx_t n = ncols(A);
    const T zero(0);
    const T ten(10);

    tlapack_check(nrows(A) == n);
    tlapack_check(nrows(Q) == n);
    tlapack_check(ncols(Q) == n);
    tlapack_check(0 <= j0);
    tlapack_check(j0 + n1 + n2 <= n);
    tlapack_check(n1 == 1 or n1 == 2);
    tlapack_check(n2 == 1 or n2 == 2);

    const idx_t j1 = j0 + 1;
    const idx_t j2 = j0 + 2;
    const idx_t j3 = j0 + 3;

    // Check if the 2x2 eigenvalue blocks consist of 2 1x1 blocks
    // If so, treat them separately
    if (n1 == 2)
        if (A(j1, j0) == zero) {
            // only 2x2 swaps can fail, so we don't need to check for error
            generalized_schur_swap(want_q, want_z, A, B, Q, Z, j1, (idx_t)1,
                                   n2);
            generalized_schur_swap(want_q, want_z, A, B, Q, Z, j0, (idx_t)1,
                                   n2);
            return 0;
        }
    if (n2 == 2)
        if (A(j0 + n1 + 1, j0 + n1) == zero) {
            // only 2x2 swaps can fail, so we don't need to check for error
            generalized_schur_swap(want_q, want_z, A, B, Q, Z, j0, n1,
                                   (idx_t)1);
            generalized_schur_swap(want_q, want_z, A, B, Q, Z, j1, n1,
                                   (idx_t)1);
            return 0;
        }

    if (n1 == 1 and n2 == 1) {
        //
        // Swap two 1-by-1 blocks.
        //
        const T a00 = A(j0, j0);
        const T a01 = A(j0, j1);
        const T a11 = A(j1, j1);
        const T b00 = B(j0, j0);
        const T b01 = B(j0, j1);
        const T b11 = B(j1, j1);

        const bool use_b = abs(b11 * a00) > abs(b00 * a11);

        //
        // Determine the transformation to perform the interchange
        //
        T cl, sl, cr, sr;
        T temp = b11 * a00 - a11 * b00;
        T temp2 = b11 * a01 - a11 * b01;
        rotg(temp2, temp, cr, sr);

        // Apply transformation from the right
        {
            auto a1 = slice(A, range{0, j1 + 1}, j0);
            auto a2 = slice(A, range{0, j1 + 1}, j1);
            rot(a2, a1, cr, sr);
            auto b1 = slice(B, range{0, j1 + 1}, j0);
            auto b2 = slice(B, range{0, j1 + 1}, j1);
            rot(b2, b1, cr, sr);
            if (want_z) {
                auto z1 = col(Z, j0);
                auto z2 = col(Z, j1);
                rot(z2, z1, cr, sr);
            }
        }

        if (use_b) {
            temp = B(j0, j0);
            temp2 = B(j1, j0);
        }
        else {
            temp = A(j0, j0);
            temp2 = A(j1, j0);
        }
        rotg(temp, temp2, cl, sl);

        // Apply transformation from the left
        {
            auto a1 = slice(A, j0, range{j0, n});
            auto a2 = slice(A, j1, range{j0, n});
            rot(a1, a2, cl, sl);
            auto b1 = slice(B, j0, range{j0, n});
            auto b2 = slice(B, j1, range{j0, n});
            rot(b1, b2, cl, sl);
            if (want_q) {
                auto q1 = col(Q, j0);
                auto q2 = col(Q, j1);
                rot(q1, q2, cl, sl);
            }
        }

        A(j1, j0) = (T)0;
        B(j1, j0) = (T)0;
    }
    if (n1 == 1 and n2 == 2) {
        //
        // Swap 1-by-1 block with 2-by-2 block
        //

        // TODO
    }
    if (n1 == 2 and n2 == 1) {
        //
        // Swap 2-by-2 block with 1-by-1 block
        //

        // TODO
    }
    if (n1 == 2 and n2 == 2) {
        //
        // Swap 2-by-2 block with 2-by-2 block
        //

        // TODO
    }

    // Standardize the 2x2 Schur blocks (if any)
    if (n2 == 2) {
        // TODO
    }
    if (n1 == 2) {
        // TODO
    }

    return 0;
}

/** schur_swap, swaps 2 eigenvalues of A.
 *
 * Implementation for complex matrices
 *
 * @ingroup auxiliary
 */
template <TLAPACK_CSMATRIX matrix_t,
          enable_if_t<is_complex<type_t<matrix_t>>, bool> = true>
int generalized_schur_swap(bool want_q,
                           bool want_z,
                           matrix_t& A,
                           matrix_t& B,
                           matrix_t& Q,
                           matrix_t& Z,
                           const size_type<matrix_t>& j0,
                           const size_type<matrix_t>& n1,
                           const size_type<matrix_t>& n2)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = ncols(A);

    tlapack_check(nrows(A) == n);
    tlapack_check(nrows(Q) == n);
    tlapack_check(ncols(Q) == n);
    tlapack_check(0 <= j0 and j0 < n);
    tlapack_check(n1 == 1);
    tlapack_check(n2 == 1);

    const idx_t j1 = j0 + 1;

    //
    // In the complex case, there can only be 1x1 blocks to swap
    //
    const T a00 = A(j0, j0);
    const T a01 = A(j0, j1);
    const T a11 = A(j1, j1);
    const T b00 = B(j0, j0);
    const T b01 = B(j0, j1);
    const T b11 = B(j1, j1);

    const bool use_b = abs(b11 * a00) > abs(b00 * a11);

    //
    // Determine the transformation to perform the interchange
    //
    real_t cl, cr;
    T sl, sr;
    T temp = b11 * a00 - a11 * b00;
    T temp2 = b11 * a01 - a11 * b01;
    rotg(temp2, temp, cr, sr);

    // Apply transformation from the right
    {
        auto a1 = slice(A, range{0, j1 + 1}, j0);
        auto a2 = slice(A, range{0, j1 + 1}, j1);
        rot(a2, a1, cr, sr);
        auto b1 = slice(B, range{0, j1 + 1}, j0);
        auto b2 = slice(B, range{0, j1 + 1}, j1);
        rot(b2, b1, cr, sr);
        if (want_z) {
            auto z1 = col(Z, j0);
            auto z2 = col(Z, j1);
            rot(z2, z1, cr, sr);
        }
    }

    if (use_b) {
        temp = B(j0, j0);
        temp2 = B(j1, j0);
    }
    else {
        temp = A(j0, j0);
        temp2 = A(j1, j0);
    }
    rotg(temp, temp2, cl, sl);

    // Apply transformation from the left
    {
        auto a1 = slice(A, j0, range{j0, n});
        auto a2 = slice(A, j1, range{j0, n});
        rot(a1, a2, cl, sl);
        auto b1 = slice(B, j0, range{j0, n});
        auto b2 = slice(B, j1, range{j0, n});
        rot(b1, b2, cl, sl);
        if (want_q) {
            auto q1 = col(Q, j0);
            auto q2 = col(Q, j1);
            rot(q1, q2, cl, conj(sl));
        }
    }

    A(j1, j0) = (T)0;
    B(j1, j0) = (T)0;

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GENERALIZED_SCHUR_SWAP_HH
