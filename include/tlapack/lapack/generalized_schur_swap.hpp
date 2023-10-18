/// @file generalized_schur_swap.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dtgex2.f
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
#include "tlapack/blas/trsv.hpp"
#include "tlapack/lapack/getrf.hpp"
#include "tlapack/lapack/inv_house3.hpp"
#include "tlapack/lapack/lahqz_eig22.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/svd22.hpp"

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

        std::vector<T> H_;
        auto H = new_matrix(H_, 2, 3);
        std::vector<T> v(3);

        complex_type<T> alpha1, alpha2;
        T beta1, beta2;

        auto A1 = slice(A, range(j1, j3), range(j1, j3));
        auto B1 = slice(B, range(j1, j3), range(j1, j3));
        lahqz_eig22(A1, B1, alpha1, alpha2, beta1, beta2);
        auto a00 = A(j0, j0);
        auto b00 = B(j0, j0);

        bool use_b = abs(b00 * alpha1) > abs(beta1 * a00);

        H(0, 0) = b00 * A(j2, j1) - a00 * B(j2, j1);
        H(0, 1) = b00 * A(j1, j1) - a00 * B(j1, j1);
        H(0, 2) = b00 * A(j0, j1) - a00 * B(j0, j1);
        H(1, 0) = b00 * A(j2, j2) - a00 * B(j2, j2);
        H(1, 1) = b00 * A(j1, j2) - a00 * B(j1, j2);
        H(1, 2) = b00 * A(j0, j2) - a00 * B(j0, j2);

        T tau;
        inv_house3(H, v, tau);

        // Apply update from the left
        for (idx_t j = j0; j < n; ++j) {
            T sum = A(j2, j) + v[1] * A(j1, j) + v[2] * A(j0, j);
            A(j2, j) = A(j2, j) - sum * tau;
            A(j1, j) = A(j1, j) - sum * tau * v[1];
            A(j0, j) = A(j0, j) - sum * tau * v[2];
        }
        for (idx_t j = j0; j < n; ++j) {
            T sum = B(j2, j) + v[1] * B(j1, j) + v[2] * B(j0, j);
            B(j2, j) = B(j2, j) - sum * tau;
            B(j1, j) = B(j1, j) - sum * tau * v[1];
            B(j0, j) = B(j0, j) - sum * tau * v[2];
        }
        for (idx_t j = 0; j < n; ++j) {
            T sum = Q(j, j2) + v[1] * Q(j, j1) + v[2] * Q(j, j0);
            Q(j, j2) = Q(j, j2) - sum * tau;
            Q(j, j1) = Q(j, j1) - sum * tau * v[1];
            Q(j, j0) = Q(j, j0) - sum * tau * v[2];
        }

        if (use_b) {
            v[0] = B(j2, j2);
            v[1] = B(j2, j1);
            v[2] = B(j2, j0);
        }
        else {
            v[0] = A(j2, j2);
            v[1] = A(j2, j1);
            v[2] = A(j2, j0);
        }

        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau);

        // Apply update from the right
        for (idx_t j = 0; j < j3; ++j) {
            T sum = A(j, j2) + v[1] * A(j, j1) + v[2] * A(j, j0);
            A(j, j2) = A(j, j2) - sum * tau;
            A(j, j1) = A(j, j1) - sum * tau * v[1];
            A(j, j0) = A(j, j0) - sum * tau * v[2];
        }
        for (idx_t j = 0; j < j3; ++j) {
            T sum = B(j, j2) + v[1] * B(j, j1) + v[2] * B(j, j0);
            B(j, j2) = B(j, j2) - sum * tau;
            B(j, j1) = B(j, j1) - sum * tau * v[1];
            B(j, j0) = B(j, j0) - sum * tau * v[2];
        }
        for (idx_t j = 0; j < n; ++j) {
            T sum = Z(j, j2) + v[1] * Z(j, j1) + v[2] * Z(j, j0);
            Z(j, j2) = Z(j, j2) - sum * tau;
            Z(j, j1) = Z(j, j1) - sum * tau * v[1];
            Z(j, j0) = Z(j, j0) - sum * tau * v[2];
        }

        A(j2, j0) = (T)0;
        A(j2, j1) = (T)0;
        B(j2, j0) = (T)0;
        B(j2, j1) = (T)0;
    }
    if (n1 == 2 and n2 == 1) {
        //
        // Swap 2-by-2 block with 1-by-1 block
        //
        std::vector<T> H_;
        auto H = new_matrix(H_, 2, 3);
        std::vector<T> v(3);

        complex_type<T> alpha1, alpha2;
        T beta1, beta2;

        auto A1 = slice(A, range(j0, j2), range(j0, j2));
        auto B1 = slice(B, range(j0, j2), range(j0, j2));
        lahqz_eig22(A1, B1, alpha1, alpha2, beta1, beta2);
        auto a22 = A(j2, j2);
        auto b22 = B(j2, j2);

        bool use_b = abs(b22 * alpha1) > abs(beta1 * a22);

        H(0, 0) = b22 * A(j0, j0) - a22 * B(j0, j0);
        H(0, 1) = b22 * A(j0, j1) - a22 * B(j0, j1);
        H(0, 2) = b22 * A(j0, j2) - a22 * B(j0, j2);
        H(1, 0) = b22 * A(j1, j0) - a22 * B(j1, j0);
        H(1, 1) = b22 * A(j1, j1) - a22 * B(j1, j1);
        H(1, 2) = b22 * A(j1, j2) - a22 * B(j1, j2);

        T tau;
        inv_house3(H, v, tau);

        // Apply update from the right
        for (idx_t j = 0; j < j3; ++j) {
            T sum = A(j, j0) + v[1] * A(j, j1) + v[2] * A(j, j2);
            A(j, j0) = A(j, j0) - sum * tau;
            A(j, j1) = A(j, j1) - sum * tau * v[1];
            A(j, j2) = A(j, j2) - sum * tau * v[2];
        }
        for (idx_t j = 0; j < j3; ++j) {
            T sum = B(j, j0) + v[1] * B(j, j1) + v[2] * B(j, j2);
            B(j, j0) = B(j, j0) - sum * tau;
            B(j, j1) = B(j, j1) - sum * tau * v[1];
            B(j, j2) = B(j, j2) - sum * tau * v[2];
        }
        for (idx_t j = 0; j < n; ++j) {
            T sum = Z(j, j0) + v[1] * Z(j, j1) + v[2] * Z(j, j2);
            Z(j, j0) = Z(j, j0) - sum * tau;
            Z(j, j1) = Z(j, j1) - sum * tau * v[1];
            Z(j, j2) = Z(j, j2) - sum * tau * v[2];
        }

        if (use_b) {
            v[0] = B(j0, j0);
            v[1] = B(j1, j0);
            v[2] = B(j2, j0);
        }
        else {
            v[0] = A(j0, j0);
            v[1] = A(j1, j0);
            v[2] = A(j2, j0);
        }
        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau);

        // Apply update from the left
        for (idx_t j = j0; j < n; ++j) {
            T sum = A(j0, j) + v[1] * A(j1, j) + v[2] * A(j2, j);
            A(j0, j) = A(j0, j) - sum * tau;
            A(j1, j) = A(j1, j) - sum * tau * v[1];
            A(j2, j) = A(j2, j) - sum * tau * v[2];
        }
        for (idx_t j = j0; j < n; ++j) {
            T sum = B(j0, j) + v[1] * B(j1, j) + v[2] * B(j2, j);
            B(j0, j) = B(j0, j) - sum * tau;
            B(j1, j) = B(j1, j) - sum * tau * v[1];
            B(j2, j) = B(j2, j) - sum * tau * v[2];
        }
        for (idx_t j = 0; j < n; ++j) {
            T sum = Q(j, j0) + v[1] * Q(j, j1) + v[2] * Q(j, j2);
            Q(j, j0) = Q(j, j0) - sum * tau;
            Q(j, j1) = Q(j, j1) - sum * tau * v[1];
            Q(j, j2) = Q(j, j2) - sum * tau * v[2];
        }

        A(j1, j0) = (T)0;
        A(j2, j0) = (T)0;
        B(j1, j0) = (T)0;
        B(j2, j0) = (T)0;
    }
    if (n1 == 2 and n2 == 2) {
        //
        // Swap 2-by-2 block with 2-by-2 block
        //
        std::vector<T> M_;
        auto M = new_matrix(M_, 8, 8);
        std::vector<T> x(8);
        std::vector<idx_t> piv(8);

        for (idx_t j = 0; j < 8; ++j)
            for (idx_t i = 0; i < 8; ++i)
                M(i, j) = (T)0;

        // Construct matrix with kronecker structure
        // I (x) A00
        M(0, 0) = A(j0, j0);
        M(0, 1) = A(j0, j1);
        M(1, 0) = A(j1, j0);
        M(1, 1) = A(j1, j1);
        M(2, 2) = A(j0, j0);
        M(2, 3) = A(j0, j1);
        M(3, 2) = A(j1, j0);
        M(3, 3) = A(j1, j1);
        // I (x) B00
        M(4, 0) = B(j0, j0);
        M(4, 1) = B(j0, j1);
        M(5, 0) = B(j1, j0);
        M(5, 1) = B(j1, j1);
        M(6, 2) = B(j0, j0);
        M(6, 3) = B(j0, j1);
        M(7, 2) = B(j1, j0);
        M(7, 3) = B(j1, j1);
        // A11T (x) I
        M(0, 4) = -A(j2, j2);
        M(0, 5) = -A(j3, j2);
        M(1, 6) = -A(j2, j2);
        M(1, 7) = -A(j3, j2);
        M(2, 4) = -A(j2, j3);
        M(2, 5) = -A(j3, j3);
        M(3, 6) = -A(j2, j3);
        M(3, 7) = -A(j3, j3);
        // B11T (x) I
        M(4, 4) = -B(j2, j2);
        M(4, 5) = -B(j3, j2);
        M(5, 6) = -B(j2, j2);
        M(5, 7) = -B(j3, j2);
        M(6, 4) = -B(j2, j3);
        M(6, 5) = -B(j3, j3);
        M(7, 6) = -B(j2, j3);
        M(7, 7) = -B(j3, j3);
        // RHS
        x[0] = A(j0, j2);
        x[1] = A(j1, j2);
        x[2] = A(j0, j3);
        x[3] = A(j1, j3);
        x[4] = B(j0, j2);
        x[5] = B(j1, j2);
        x[6] = B(j0, j3);
        x[7] = B(j1, j3);
        // LU of M
        int ierr = getrf(M, piv);
        if (ierr != 0) return 1;
        // Apply pivot to rhs
        for (idx_t i = 0; i < 8; ++i) {
            if (i != piv[i]) std::swap(x[i], x[piv[i]]);
        }
        // Solve Ly = rhs
        trsv(LOWER_TRIANGLE, NO_TRANS, UNIT_DIAG, M, x);
        // Solve Ux = y
        trsv(UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, M, x);

        // Find Q so that
        //       [ -x[0] -x[2] ]   [ *  * ]
        //  Q^T  [ -x[1] -x[3] ] = [ *  * ]
        //       [ 1     0     ]   [ 0  0 ]
        //       [ 0     1     ]   [ 0  0 ]

        // Rotation to make X upper triangular
        T cxl1, sxl1;
        rotg(x[0], x[1], cxl1, sxl1);
        x[1] = (T)0;
        T rottemp = cxl1 * x[2] + sxl1 * x[3];
        x[3] = -sxl1 * x[2] + cxl1 * x[3];
        x[2] = rottemp;
        // SVD of (upper triangular) X
        T cxl2, sxl2, cxr, sxr, ssx1, ssx2;
        svd22(x[0], x[2], x[3], ssx2, ssx1, cxl2, sxl2, cxr, sxr);
        // Fuse left rotations
        T cxl, sxl;
        cxl = cxl1 * cxl2 - sxl1 * sxl2;
        sxl = cxl2 * sxl1 + sxl2 * cxl1;
        // Rotations based on the singular values
        ssx1 = -ssx1;
        ssx2 = -ssx2;
        T temp = (T)1;
        T cx1, sx1, cx2, sx2;
        rotg(ssx1, temp, cx1, sx1);
        temp = (T)1;
        rotg(ssx2, temp, cx2, sx2);

        // Find Z so that
        //       [ 1     0     ]   [ 0  0 ]
        //  Z^T  [ 0     1     ] = [ 0  0 ]
        //       [ x[4]  x[6]  ]   [ *  * ]
        //       [ x[5]  x[7]  ]   [ *  * ]

        // Rotation to make Y^T upper triangular
        T cyl1, syl1;
        rotg(x[4], x[5], cyl1, syl1);
        x[5] = (T)0;
        rottemp = cyl1 * x[6] + syl1 * x[7];
        x[7] = -syl1 * x[6] + cyl1 * x[7];
        x[6] = rottemp;
        // SVD of (upper triangular) Y
        T cyl2, syl2, cyr, syr, ssy1, ssy2;
        svd22(x[4], x[6], x[7], ssy2, ssy1, cyl2, syl2, cyr, syr);
        // Fuse left rotations
        T cyl, syl;
        cyl = cyl1 * cyl2 - syl1 * syl2;
        syl = cyl2 * syl1 + syl2 * cyl1;
        // Rotations based on the singular values
        temp = (T)1;
        T cy1, sy1, cy2, sy2;
        rotg(ssy1, temp, cy1, sy1);
        temp = (T)1;
        rotg(ssy2, temp, cy2, sy2);

        // Apply rotations from the left
        {
            auto a0 = slice(A, j0, range(j0, n));
            auto a1 = slice(A, j1, range(j0, n));
            auto a2 = slice(A, j2, range(j0, n));
            auto a3 = slice(A, j3, range(j0, n));
            rot(a0, a1, cyr, syr);
            rot(a2, a3, cyl, syl);
            rot(a2, a0, cy1, sy1);
            rot(a3, a1, cy2, sy2);

            auto b0 = slice(B, j0, range(j0, n));
            auto b1 = slice(B, j1, range(j0, n));
            auto b2 = slice(B, j2, range(j0, n));
            auto b3 = slice(B, j3, range(j0, n));
            rot(b0, b1, cyr, syr);
            rot(b2, b3, cyl, syl);
            rot(b2, b0, cy1, sy1);
            rot(b3, b1, cy2, sy2);

            auto q0 = col(Q, j0);
            auto q1 = col(Q, j1);
            auto q2 = col(Q, j2);
            auto q3 = col(Q, j3);
            rot(q0, q1, cyr, syr);
            rot(q2, q3, cyl, syl);
            rot(q2, q0, cy1, sy1);
            rot(q3, q1, cy2, sy2);
        }

        // Apply rotations from the right
        {
            auto a0 = slice(A, range(0, j3 + 1), j0);
            auto a1 = slice(A, range(0, j3 + 1), j1);
            auto a2 = slice(A, range(0, j3 + 1), j2);
            auto a3 = slice(A, range(0, j3 + 1), j3);
            rot(a0, a1, cxl, sxl);
            rot(a2, a3, cxr, sxr);
            rot(a0, a2, cx1, sx1);
            rot(a1, a3, cx2, sx2);

            auto b0 = slice(B, range(0, j3 + 1), j0);
            auto b1 = slice(B, range(0, j3 + 1), j1);
            auto b2 = slice(B, range(0, j3 + 1), j2);
            auto b3 = slice(B, range(0, j3 + 1), j3);
            rot(b0, b1, cxl, sxl);
            rot(b2, b3, cxr, sxr);
            rot(b0, b2, cx1, sx1);
            rot(b1, b3, cx2, sx2);

            auto z0 = col(Z, j0);
            auto z1 = col(Z, j1);
            auto z2 = col(Z, j2);
            auto z3 = col(Z, j3);
            rot(z0, z1, cxl, sxl);
            rot(z2, z3, cxr, sxr);
            rot(z0, z2, cx1, sx1);
            rot(z1, z3, cx2, sx2);
        }

        // TODO: check backward error
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
