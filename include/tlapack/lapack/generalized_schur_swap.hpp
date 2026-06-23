/// @file generalized_schur_swap.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dtgex2.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GENERALIZED_SCHUR_SWAP_HH
#define TLAPACK_GENERALIZED_SCHUR_SWAP_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
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
 *                Unitary matrix, not referenced if want_q is false
 * @param[in,out] Z n-by-n matrix.
 *                Unitary matrix, not referenced if want_z is false
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
            int info;
            info = generalized_schur_swap(want_q, want_z, A, B, Q, Z, j1,
                                          (idx_t)1, n2);
            if (info != 0) return info;
            info = generalized_schur_swap(want_q, want_z, A, B, Q, Z, j0,
                                          (idx_t)1, n2);
            if (info != 0) return info;
            return 0;
        }
    if (n2 == 2)
        if (A(j0 + n1 + 1, j0 + n1) == zero) {
            int info;
            info = generalized_schur_swap(want_q, want_z, A, B, Q, Z, j0, n1,
                                          (idx_t)1);
            if (info != 0) return info;
            info = generalized_schur_swap(want_q, want_z, A, B, Q, Z, j1, n1,
                                          (idx_t)1);
            if (info != 0) return info;
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
        std::vector<T> vl(3);
        std::vector<T> vrA(3);
        std::vector<T> vrB(3);
        T taul, taurA, taurB;

        std::vector<T> AA_;
        auto AA = new_matrix(AA_, 3, 3);
        std::vector<T> BB_;
        auto BB = new_matrix(BB_, 3, 3);
        lacpy(GENERAL, slice(A, range(j0, j3), range(j0, j3)), AA);
        lacpy(GENERAL, slice(B, range(j0, j3), range(j0, j3)), BB);

        auto a00 = AA(0, 0);
        auto b00 = BB(0, 0);

        T norma = lange(FROB_NORM, AA);
        T normb = lange(FROB_NORM, BB);

        H(0, 0) = b00 * AA(2, 1) - a00 * BB(2, 1);
        H(0, 1) = b00 * AA(1, 1) - a00 * BB(1, 1);
        H(0, 2) = b00 * AA(0, 1) - a00 * BB(0, 1);
        H(1, 0) = b00 * AA(2, 2) - a00 * BB(2, 2);
        H(1, 1) = b00 * AA(1, 2) - a00 * BB(1, 2);
        H(1, 2) = b00 * AA(0, 2) - a00 * BB(0, 2);

        inv_house3(H, vl, taul);

        // Tentatively apply update from the left to local matrices
        for (idx_t j = 0; j < 3; ++j) {
            T sum = AA(2, j) + vl[1] * AA(1, j) + vl[2] * AA(0, j);
            AA(2, j) = AA(2, j) - sum * taul;
            AA(1, j) = AA(1, j) - sum * taul * vl[1];
            AA(0, j) = AA(0, j) - sum * taul * vl[2];
        }
        for (idx_t j = 0; j < 3; ++j) {
            T sum = BB(2, j) + vl[1] * BB(1, j) + vl[2] * BB(0, j);
            BB(2, j) = BB(2, j) - sum * taul;
            BB(1, j) = BB(1, j) - sum * taul * vl[1];
            BB(0, j) = BB(0, j) - sum * taul * vl[2];
        }

        // Determine two sets of right transformations
        // one that zeroes out the last row of A and one that zeroes out the
        // last row of B
        // We will later choose the one that gives the smaller error
        vrA[0] = AA(2, 2);
        vrA[1] = AA(2, 1);
        vrA[2] = AA(2, 0);
        larfg(FORWARD, COLUMNWISE_STORAGE, vrA, taurA);

        vrB[0] = BB(2, 2);
        vrB[1] = BB(2, 1);
        vrB[2] = BB(2, 0);
        larfg(FORWARD, COLUMNWISE_STORAGE, vrB, taurB);

        // Apply the update calculated using BB to AA
        // and vice versa. We choose the one that gives the smaller error
        for (idx_t j = 2; j < 3; ++j) {
            T sum = AA(j, 2) + vrB[1] * AA(j, 1) + vrB[2] * AA(j, 0);
            AA(j, 2) = AA(j, 2) - sum * taurB;
            AA(j, 1) = AA(j, 1) - sum * taurB * vrB[1];
            AA(j, 0) = AA(j, 0) - sum * taurB * vrB[2];
        }
        for (idx_t j = 2; j < 3; ++j) {
            T sum = BB(j, 2) + vrA[1] * BB(j, 1) + vrA[2] * BB(j, 0);
            BB(j, 2) = BB(j, 2) - sum * taurA;
            BB(j, 1) = BB(j, 1) - sum * taurA * vrA[1];
            BB(j, 0) = BB(j, 0) - sum * taurA * vrA[2];
        }

        //
        // Determine if the swap was successful
        //
        T errA = lapy2(AA(2, 0), AA(2, 1));
        T errB = lapy2(BB(2, 0), BB(2, 1));
        const T eps = ulp<T>();
        const T small_num = safe_min<T>();

        if (errA > max((T)20 * norma * eps, small_num) and
            errB > max((T)20 * normb * eps, small_num)) {
            // The swap failed, return with error
            // Note, though we don't have a proof that this will always be the
            // case, there are currently no known cases where this swap can
            // fail.
            return 1;
        }

        //
        // Swap is accepted, apply the updates to the original matrices
        //
        for (idx_t j = j0; j < n; ++j) {
            T sum = A(j2, j) + vl[1] * A(j1, j) + vl[2] * A(j0, j);
            A(j2, j) = A(j2, j) - sum * taul;
            A(j1, j) = A(j1, j) - sum * taul * vl[1];
            A(j0, j) = A(j0, j) - sum * taul * vl[2];
        }
        for (idx_t j = j0; j < n; ++j) {
            T sum = B(j2, j) + vl[1] * B(j1, j) + vl[2] * B(j0, j);
            B(j2, j) = B(j2, j) - sum * taul;
            B(j1, j) = B(j1, j) - sum * taul * vl[1];
            B(j0, j) = B(j0, j) - sum * taul * vl[2];
        }
        if (want_q) {
            for (idx_t j = 0; j < n; ++j) {
                T sum = Q(j, j2) + vl[1] * Q(j, j1) + vl[2] * Q(j, j0);
                Q(j, j2) = Q(j, j2) - sum * taul;
                Q(j, j1) = Q(j, j1) - sum * taul * vl[1];
                Q(j, j0) = Q(j, j0) - sum * taul * vl[2];
            }
        }
        if (errB * norma < errA * normb) {
            // The error is smallest when using vrA
            for (idx_t j = 0; j < j3; ++j) {
                T sum = A(j, j2) + vrA[1] * A(j, j1) + vrA[2] * A(j, j0);
                A(j, j2) = A(j, j2) - sum * taurA;
                A(j, j1) = A(j, j1) - sum * taurA * vrA[1];
                A(j, j0) = A(j, j0) - sum * taurA * vrA[2];
            }
            for (idx_t j = 0; j < j3; ++j) {
                T sum = B(j, j2) + vrA[1] * B(j, j1) + vrA[2] * B(j, j0);
                B(j, j2) = B(j, j2) - sum * taurA;
                B(j, j1) = B(j, j1) - sum * taurA * vrA[1];
                B(j, j0) = B(j, j0) - sum * taurA * vrA[2];
            }
            if (want_z) {
                for (idx_t j = 0; j < n; ++j) {
                    T sum = Z(j, j2) + vrA[1] * Z(j, j1) + vrA[2] * Z(j, j0);
                    Z(j, j2) = Z(j, j2) - sum * taurA;
                    Z(j, j1) = Z(j, j1) - sum * taurA * vrA[1];
                    Z(j, j0) = Z(j, j0) - sum * taurA * vrA[2];
                }
            }
        }
        else {
            // The error is smallest when using vrB
            for (idx_t j = 0; j < j3; ++j) {
                T sum = A(j, j2) + vrB[1] * A(j, j1) + vrB[2] * A(j, j0);
                A(j, j2) = A(j, j2) - sum * taurB;
                A(j, j1) = A(j, j1) - sum * taurB * vrB[1];
                A(j, j0) = A(j, j0) - sum * taurB * vrB[2];
            }
            for (idx_t j = 0; j < j3; ++j) {
                T sum = B(j, j2) + vrB[1] * B(j, j1) + vrB[2] * B(j, j0);
                B(j, j2) = B(j, j2) - sum * taurB;
                B(j, j1) = B(j, j1) - sum * taurB * vrB[1];
                B(j, j0) = B(j, j0) - sum * taurB * vrB[2];
            }
            if (want_z) {
                for (idx_t j = 0; j < n; ++j) {
                    T sum = Z(j, j2) + vrB[1] * Z(j, j1) + vrB[2] * Z(j, j0);
                    Z(j, j2) = Z(j, j2) - sum * taurB;
                    Z(j, j1) = Z(j, j1) - sum * taurB * vrB[1];
                    Z(j, j0) = Z(j, j0) - sum * taurB * vrB[2];
                }
            }
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
        T taur, taulA, taulB;
        std::vector<T> vr(3);
        std::vector<T> vlA(3);
        std::vector<T> vlB(3);

        std::vector<T> AA_;
        auto AA = new_matrix(AA_, 3, 3);
        std::vector<T> BB_;
        auto BB = new_matrix(BB_, 3, 3);
        lacpy(GENERAL, slice(A, range(j0, j3), range(j0, j3)), AA);
        lacpy(GENERAL, slice(B, range(j0, j3), range(j0, j3)), BB);

        auto a22 = A(j2, j2);
        auto b22 = B(j2, j2);

        T norma = lange(FROB_NORM, AA);
        T normb = lange(FROB_NORM, BB);

        H(0, 0) = b22 * A(j0, j0) - a22 * B(j0, j0);
        H(0, 1) = b22 * A(j0, j1) - a22 * B(j0, j1);
        H(0, 2) = b22 * A(j0, j2) - a22 * B(j0, j2);
        H(1, 0) = b22 * A(j1, j0) - a22 * B(j1, j0);
        H(1, 1) = b22 * A(j1, j1) - a22 * B(j1, j1);
        H(1, 2) = b22 * A(j1, j2) - a22 * B(j1, j2);

        inv_house3(H, vr, taur);

        // Apply update from the right to the local matrices
        for (idx_t j = 0; j < 3; ++j) {
            T sum = AA(j, 0) + vr[1] * AA(j, 1) + vr[2] * AA(j, 2);
            AA(j, 0) = AA(j, 0) - sum * taur;
            AA(j, 1) = AA(j, 1) - sum * taur * vr[1];
            AA(j, 2) = AA(j, 2) - sum * taur * vr[2];
        }
        for (idx_t j = 0; j < 3; ++j) {
            T sum = BB(j, 0) + vr[1] * BB(j, 1) + vr[2] * BB(j, 2);
            BB(j, 0) = BB(j, 0) - sum * taur;
            BB(j, 1) = BB(j, 1) - sum * taur * vr[1];
            BB(j, 2) = BB(j, 2) - sum * taur * vr[2];
        }

        // Determine two sets of left transformations
        // one that zeroes out the first column of A and one that zeroes out the
        // first column of B
        // We will later choose the one that gives the smaller error
        vlA[0] = AA(0, 0);
        vlA[1] = AA(1, 0);
        vlA[2] = AA(2, 0);
        larfg(FORWARD, COLUMNWISE_STORAGE, vlA, taulA);

        vlB[0] = BB(0, 0);
        vlB[1] = BB(1, 0);
        vlB[2] = BB(2, 0);
        larfg(FORWARD, COLUMNWISE_STORAGE, vlB, taulB);

        // Apply update from the left
        for (idx_t j = 0; j < 3; ++j) {
            T sum = AA(0, j) + vlB[1] * AA(1, j) + vlB[2] * AA(2, j);
            AA(0, j) = AA(0, j) - sum * taulB;
            AA(1, j) = AA(1, j) - sum * taulB * vlB[1];
            AA(2, j) = AA(2, j) - sum * taulB * vlB[2];
        }
        for (idx_t j = 0; j < 3; ++j) {
            T sum = BB(0, j) + vlA[1] * BB(1, j) + vlA[2] * BB(2, j);
            BB(0, j) = BB(0, j) - sum * taulA;
            BB(1, j) = BB(1, j) - sum * taulA * vlA[1];
            BB(2, j) = BB(2, j) - sum * taulA * vlA[2];
        }

        //
        // Determine if the swap was successful
        //
        T errA = lapy2(AA(1, 0), AA(2, 0));
        T errB = lapy2(BB(1, 0), BB(2, 0));
        const T eps = ulp<T>();
        const T small_num = safe_min<T>();

        if (errA > max((T)20 * norma * eps, small_num) and
            errB > max((T)20 * normb * eps, small_num)) {
            // The swap failed, return with error
            // Note, though we don't have a proof that this will always be the
            // case, there are currently no known cases where this swap can
            // fail.
            return 1;
        }

        //
        // Swap is accepted, apply the updates to the original matrices
        //
        for (idx_t j = 0; j < j3; ++j) {
            T sum = A(j, j0) + vr[1] * A(j, j1) + vr[2] * A(j, j2);
            A(j, j0) = A(j, j0) - sum * taur;
            A(j, j1) = A(j, j1) - sum * taur * vr[1];
            A(j, j2) = A(j, j2) - sum * taur * vr[2];
        }
        for (idx_t j = 0; j < j3; ++j) {
            T sum = B(j, j0) + vr[1] * B(j, j1) + vr[2] * B(j, j2);
            B(j, j0) = B(j, j0) - sum * taur;
            B(j, j1) = B(j, j1) - sum * taur * vr[1];
            B(j, j2) = B(j, j2) - sum * taur * vr[2];
        }
        if (want_z) {
            for (idx_t j = 0; j < n; ++j) {
                T sum = Z(j, j0) + vr[1] * Z(j, j1) + vr[2] * Z(j, j2);
                Z(j, j0) = Z(j, j0) - sum * taur;
                Z(j, j1) = Z(j, j1) - sum * taur * vr[1];
                Z(j, j2) = Z(j, j2) - sum * taur * vr[2];
            }
        }
        if (errB * norma < errA * normb) {
            // The error is smallest when using vlA
            for (idx_t j = j0; j < n; ++j) {
                T sum = A(j0, j) + vlA[1] * A(j1, j) + vlA[2] * A(j2, j);
                A(j0, j) = A(j0, j) - sum * taulA;
                A(j1, j) = A(j1, j) - sum * taulA * vlA[1];
                A(j2, j) = A(j2, j) - sum * taulA * vlA[2];
            }
            for (idx_t j = j0; j < n; ++j) {
                T sum = B(j0, j) + vlA[1] * B(j1, j) + vlA[2] * B(j2, j);
                B(j0, j) = B(j0, j) - sum * taulA;
                B(j1, j) = B(j1, j) - sum * taulA * vlA[1];
                B(j2, j) = B(j2, j) - sum * taulA * vlA[2];
            }
            if (want_q) {
                for (idx_t j = 0; j < n; ++j) {
                    T sum = Q(j, j0) + vlA[1] * Q(j, j1) + vlA[2] * Q(j, j2);
                    Q(j, j0) = Q(j, j0) - sum * taulA;
                    Q(j, j1) = Q(j, j1) - sum * taulA * vlA[1];
                    Q(j, j2) = Q(j, j2) - sum * taulA * vlA[2];
                }
            }
        }
        else {
            // The error is smallest when using vlB
            for (idx_t j = j0; j < n; ++j) {
                T sum = A(j0, j) + vlB[1] * A(j1, j) + vlB[2] * A(j2, j);
                A(j0, j) = A(j0, j) - sum * taulB;
                A(j1, j) = A(j1, j) - sum * taulB * vlB[1];
                A(j2, j) = A(j2, j) - sum * taulB * vlB[2];
            }
            for (idx_t j = j0; j < n; ++j) {
                T sum = B(j0, j) + vlB[1] * B(j1, j) + vlB[2] * B(j2, j);
                B(j0, j) = B(j0, j) - sum * taulB;
                B(j1, j) = B(j1, j) - sum * taulB * vlB[1];
                B(j2, j) = B(j2, j) - sum * taulB * vlB[2];
            }
            if (want_q) {
                for (idx_t j = 0; j < n; ++j) {
                    T sum = Q(j, j0) + vlB[1] * Q(j, j1) + vlB[2] * Q(j, j2);
                    Q(j, j0) = Q(j, j0) - sum * taulB;
                    Q(j, j1) = Q(j, j1) - sum * taulB * vlB[1];
                    Q(j, j2) = Q(j, j2) - sum * taulB * vlB[2];
                }
            }
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
        if (ierr != 0) {
            return 1;
        }
        // Apply pivot to rhs
        for (idx_t i = 0; i < 8; ++i) {
            if (i != piv[i]) std::swap(x[i], x[piv[i]]);
        }
        // Solve Ly = rhs
        trsv(LOWER_TRIANGLE, NO_TRANS, UNIT_DIAG, M, x);
        // Solve Ux = y
        trsv(UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, M, x);

        // Find Zc so that
        //       [ -x[0] -x[2] ]   [ *  * ]
        //  Zc^T [ -x[1] -x[3] ] = [ *  * ]
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

        // Find Qc so that
        //       [ 1     0     ]   [ 0  0 ]
        //  Qc^T  [ 0     1     ] = [ 0  0 ]
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

        // Perform the swap on a local matrix and check the error
        std::vector<T> AA_;
        auto AA = new_matrix(AA_, 4, 4);
        std::vector<T> BB_;
        auto BB = new_matrix(BB_, 4, 4);
        std::vector<T> QQ_;
        auto QQ = new_matrix(QQ_, 4, 4);
        std::vector<T> ZZ_;
        auto ZZ = new_matrix(ZZ_, 4, 4);

        lacpy(GENERAL, slice(A, range(j0, j3 + 1), range(j0, j3 + 1)), AA);
        lacpy(GENERAL, slice(B, range(j0, j3 + 1), range(j0, j3 + 1)), BB);
        laset(GENERAL, T(0), T(1), QQ);
        laset(GENERAL, T(0), T(1), ZZ);

        auto norma = lange(FROB_NORM, AA);
        auto normb = lange(FROB_NORM, BB);

        // Apply rotations from the left to local matrices
        {
            auto a0 = row(AA, 0);
            auto a1 = row(AA, 1);
            auto a2 = row(AA, 2);
            auto a3 = row(AA, 3);
            rot(a0, a1, cyr, syr);
            rot(a2, a3, cyl, syl);
            rot(a2, a0, cy1, sy1);
            rot(a3, a1, cy2, sy2);

            auto b0 = row(BB, 0);
            auto b1 = row(BB, 1);
            auto b2 = row(BB, 2);
            auto b3 = row(BB, 3);
            rot(b0, b1, cyr, syr);
            rot(b2, b3, cyl, syl);
            rot(b2, b0, cy1, sy1);
            rot(b3, b1, cy2, sy2);

            auto q0 = col(QQ, 0);
            auto q1 = col(QQ, 1);
            auto q2 = col(QQ, 2);
            auto q3 = col(QQ, 3);

            rot(q0, q1, cyr, syr);
            rot(q2, q3, cyl, syl);
            rot(q2, q0, cy1, sy1);
            rot(q3, q1, cy2, sy2);
        }
        // Apply rotations from the right to local matrices
        {
            auto a0 = col(AA, 0);
            auto a1 = col(AA, 1);
            auto a2 = col(AA, 2);
            auto a3 = col(AA, 3);
            rot(a0, a1, cxl, sxl);
            rot(a2, a3, cxr, sxr);
            rot(a0, a2, cx1, sx1);
            rot(a1, a3, cx2, sx2);

            auto b0 = col(BB, 0);
            auto b1 = col(BB, 1);
            auto b2 = col(BB, 2);
            auto b3 = col(BB, 3);
            rot(b0, b1, cxl, sxl);
            rot(b2, b3, cxr, sxr);
            rot(b0, b2, cx1, sx1);
            rot(b1, b3, cx2, sx2);

            auto z0 = col(ZZ, 0);
            auto z1 = col(ZZ, 1);
            auto z2 = col(ZZ, 2);
            auto z3 = col(ZZ, 3);
            rot(z0, z1, cxl, sxl);
            rot(z2, z3, cxr, sxr);
            rot(z0, z2, cx1, sx1);
            rot(z1, z3, cx2, sx2);
        }

        // Weak stability test
        auto enorma = lange(FROB_NORM, slice(AA, range(2, 4), range(0, 2)));
        auto enormb = lange(FROB_NORM, slice(BB, range(2, 4), range(0, 2)));
        const T eps = ulp<T>();
        const T small_num = safe_min<T>();

        idx_t iter = 0;
        T tolA = max((T)20 * norma * eps, small_num);
        T tolB = max((T)20 * normb * eps, small_num);
        const idx_t max_iter = 6;
        while (iter < max_iter) {
            if (enorma <= tolA and enormb <= tolB) break;
            if (iter == max_iter - 1) {
                return 1;
            }
            // The swap is not (yet) accepted, apply iterative refinement to
            // try to improve the swap.

            for (idx_t j = 0; j < 8; ++j)
                for (idx_t i = 0; i < 8; ++i)
                    M(i, j) = (T)0;

            /// Construct matrix with kronecker structure
            // I (x) AA(2:3,2:3)
            M(0, 0) = AA(2, 2);
            M(0, 1) = AA(2, 3);
            M(1, 0) = AA(3, 2);
            M(1, 1) = AA(3, 3);
            M(2, 2) = AA(2, 2);
            M(2, 3) = AA(2, 3);
            M(3, 2) = AA(3, 2);
            M(3, 3) = AA(3, 3);
            // -AA(0:1,0:1)^T (x) I
            M(0, 4) = -AA(0, 0);
            M(0, 5) = -AA(1, 0);
            M(1, 6) = -AA(0, 0);
            M(1, 7) = -AA(1, 0);
            M(2, 4) = -AA(0, 1);
            M(2, 5) = -AA(1, 1);
            M(3, 6) = -AA(0, 1);
            M(3, 7) = -AA(1, 1);

            // I (x) BB(2:3,2:3)
            M(4, 0) = BB(2, 2);
            M(4, 1) = BB(2, 3);
            M(5, 0) = BB(3, 2);
            M(5, 1) = BB(3, 3);
            M(6, 2) = BB(2, 2);
            M(6, 3) = BB(2, 3);
            M(7, 2) = BB(3, 2);
            M(7, 3) = BB(3, 3);
            // -BB(0:1,0:1)^T (x) I
            M(4, 4) = -BB(0, 0);
            M(4, 5) = -BB(1, 0);
            M(5, 6) = -BB(0, 0);
            M(5, 7) = -BB(1, 0);
            M(6, 4) = -BB(0, 1);
            M(6, 5) = -BB(1, 1);
            M(7, 6) = -BB(0, 1);
            M(7, 7) = -BB(1, 1);

            // RHS
            x[0] = AA(2, 0);
            x[1] = AA(3, 0);
            x[2] = AA(2, 1);
            x[3] = AA(3, 1);
            x[4] = BB(2, 0);
            x[5] = BB(3, 0);
            x[6] = BB(2, 1);
            x[7] = BB(3, 1);

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

            // Find Zc so that
            //       [ 1      0    ]   [ *  * ]
            //  Zc^T [ 0      1    ] = [ *  * ]
            //       [ -x[0] -x[2] ]   [ 0  0 ]
            //       [ -x[1] -x[3] ]   [ 0  0 ]

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
            rotg(temp, ssx1, cx1, sx1);
            temp = (T)1;
            rotg(temp, ssx2, cx2, sx2);

            // Find Qc so that
            //       [ 1     0     ]   [ 0  0 ]
            //  Qc^T  [ 0     1     ] = [ 0  0 ]
            //       [ x[4]  x[6]  ]   [ *  * ]
            //       [ x[5]  x[7]  ]   [ *  * ]

            std::swap(x[5], x[6]);
            T cyl1, syl1;
            rotg(x[4], x[5], cyl1, syl1);
            x[5] = (T)0;
            rottemp = cyl1 * x[6] + syl1 * x[7];
            x[7] = -syl1 * x[6] + cyl1 * x[7];
            x[6] = rottemp;
            // SVD of (upper triangular) X
            T cyl2, syl2, cyr, syr, ssy1, ssy2;
            svd22(x[4], x[6], x[7], ssy2, ssy1, cyl2, syl2, cyr, syr);
            // Fuse left rotations
            T cyl, syl;
            cyl = cyl1 * cyl2 - syl1 * syl2;
            syl = cyl2 * syl1 + syl2 * cyl1;
            // Rotations based on the singular values
            ssy1 = -ssy1;
            ssy2 = -ssy2;
            temp = (T)1;
            T cy1, sy1, cy2, sy2;
            rotg(temp, ssy1, cy1, sy1);
            temp = (T)1;
            rotg(temp, ssy2, cy2, sy2);

            // Apply rotations from the left to local matrices
            {
                auto a0 = row(AA, 0);
                auto a1 = row(AA, 1);
                auto a2 = row(AA, 2);
                auto a3 = row(AA, 3);
                rot(a0, a1, cyr, syr);
                rot(a2, a3, cyl, syl);
                rot(a0, a2, cy1, sy1);
                rot(a1, a3, cy2, sy2);

                auto b0 = row(BB, 0);
                auto b1 = row(BB, 1);
                auto b2 = row(BB, 2);
                auto b3 = row(BB, 3);
                rot(b0, b1, cyr, syr);
                rot(b2, b3, cyl, syl);
                rot(b0, b2, cy1, sy1);
                rot(b1, b3, cy2, sy2);

                auto q0 = col(QQ, 0);
                auto q1 = col(QQ, 1);
                auto q2 = col(QQ, 2);
                auto q3 = col(QQ, 3);

                rot(q0, q1, cyr, syr);
                rot(q2, q3, cyl, syl);
                rot(q0, q2, cy1, sy1);
                rot(q1, q3, cy2, sy2);
            }
            // Apply rotations from the right to local matrices
            {
                auto a0 = col(AA, 0);
                auto a1 = col(AA, 1);
                auto a2 = col(AA, 2);
                auto a3 = col(AA, 3);
                rot(a0, a1, cxr, sxr);
                rot(a2, a3, cxl, sxl);
                rot(a0, a2, cx1, sx1);
                rot(a1, a3, cx2, sx2);

                auto b0 = col(BB, 0);
                auto b1 = col(BB, 1);
                auto b2 = col(BB, 2);
                auto b3 = col(BB, 3);
                rot(b0, b1, cxr, sxr);
                rot(b2, b3, cxl, sxl);
                rot(b0, b2, cx1, sx1);
                rot(b1, b3, cx2, sx2);

                auto z0 = col(ZZ, 0);
                auto z1 = col(ZZ, 1);
                auto z2 = col(ZZ, 2);
                auto z3 = col(ZZ, 3);
                rot(z0, z1, cxr, sxr);
                rot(z2, z3, cxl, sxl);
                rot(z0, z2, cx1, sx1);
                rot(z1, z3, cx2, sx2);
            }

            enorma = lange(FROB_NORM, slice(AA, range(2, 4), range(0, 2)));
            enormb = lange(FROB_NORM, slice(BB, range(2, 4), range(0, 2)));
            iter++;
        }

        // TODO: this is a large workspace, add it to the interface
        std::vector<T> workl_;
        auto workl = new_matrix(workl_, 4, n);
        std::vector<T> workr_;
        auto workr = new_matrix(workr_, n, 4);

        // Apply QQ
        {
            auto A_slice = slice(A, range(j0, j3 + 1), range(j0, n));
            auto work_slice = slice(workl, range(0, 4), range(j0, n));
            gemm(TRANSPOSE, NO_TRANS, (T)1, QQ, A_slice, StrongZero(T(0)),
                 work_slice);
            lacpy(GENERAL, work_slice, A_slice);
        }
        {
            auto B_slice = slice(B, range(j0, j3 + 1), range(j0, n));
            auto work_slice = slice(workl, range(0, 4), range(j0, n));
            gemm(TRANSPOSE, NO_TRANS, (T)1, QQ, B_slice, StrongZero(T(0)),
                 work_slice);
            lacpy(GENERAL, work_slice, B_slice);
        }
        {
            auto Q_slice = slice(Q, range(0, n), range(j0, j3 + 1));
            auto work_slice = slice(workr, range(0, n), range(0, 4));
            gemm(NO_TRANS, NO_TRANS, (T)1, Q_slice, QQ, StrongZero(T(0)),
                 work_slice);
            lacpy(GENERAL, work_slice, Q_slice);
        }
        // Apply ZZ
        {
            auto A_slice =
                slice(A, range(0, min(n, j3 + 1)), range(j0, j3 + 1));
            auto work_slice =
                slice(workr, range(0, min(n, j3 + 1)), range(0, 4));
            gemm(NO_TRANS, NO_TRANS, (T)1, A_slice, ZZ, StrongZero(T(0)),
                 work_slice);
            lacpy(GENERAL, work_slice, A_slice);
        }
        {
            auto B_slice =
                slice(B, range(0, min(n, j3 + 1)), range(j0, j3 + 1));
            auto work_slice =
                slice(workr, range(0, min(n, j3 + 1)), range(0, 4));
            gemm(NO_TRANS, NO_TRANS, (T)1, B_slice, ZZ, StrongZero(T(0)),
                 work_slice);
            lacpy(GENERAL, work_slice, B_slice);
        }
        {
            auto Z_slice = slice(Z, range(0, n), range(j0, j3 + 1));
            auto work_slice = slice(workr, range(0, n), range(0, 4));
            gemm(NO_TRANS, NO_TRANS, (T)1, Z_slice, ZZ, StrongZero(T(0)),
                 work_slice);
            lacpy(GENERAL, work_slice, Z_slice);
        }

        // Set relevant parts to zero
        A(j2, j0) = (T)0;
        A(j3, j0) = (T)0;
        A(j2, j1) = (T)0;
        A(j3, j1) = (T)0;
        B(j2, j0) = (T)0;
        B(j3, j0) = (T)0;
        B(j2, j1) = (T)0;
        B(j3, j1) = (T)0;
    }

    // Standardize the 2x2 Schur blocks (if any)
    if (n2 == 2) {
        // Make B upper triangular
        T cl1, sl1;
        rotg(B(j0, j0), B(j1, j0), cl1, sl1);
        B(j1, j0) = (T)0;
        {
            auto b1 = slice(B, j0, range(j1, j2));
            auto b2 = slice(B, j1, range(j1, j2));
            rot(b1, b2, cl1, sl1);
        }
        // Standard form
        T ssmin, ssmax, cl2, sl2, cr, sr;
        svd22(B(j0, j0), B(j0, j1), B(j1, j1), ssmin, ssmax, cl2, sl2, cr, sr);
        if (ssmax < (T)0) {
            cr = -cr;
            sr = -sr;
            ssmin = -ssmin;
            ssmax = -ssmax;
        }
        B(j0, j0) = ssmax;
        B(j1, j1) = ssmin;
        B(j0, j1) = (T)0;
        // Fuse left rotations
        T cl, sl;
        cl = cl1 * cl2 - sl1 * sl2;
        sl = cl2 * sl1 + sl2 * cl1;
        // Apply left rotation
        {
            auto a1 = slice(A, j0, range(j0, n));
            auto a2 = slice(A, j1, range(j0, n));
            rot(a1, a2, cl, sl);
            auto b1 = slice(B, j0, range(j2, n));
            auto b2 = slice(B, j1, range(j2, n));
            rot(b1, b2, cl, sl);
            auto q0 = col(Q, j0);
            auto q1 = col(Q, j1);
            rot(q0, q1, cl, sl);
        }
        // Apply right rotation
        {
            auto a1 = slice(A, range(0, j2), j0);
            auto a2 = slice(A, range(0, j2), j1);
            rot(a1, a2, cr, sr);
            auto b1 = slice(B, range(0, j0), j0);
            auto b2 = slice(B, range(0, j0), j1);
            rot(b1, b2, cr, sr);
            auto z0 = col(Z, j0);
            auto z1 = col(Z, j1);
            rot(z0, z1, cr, sr);
        }
    }
    if (n1 == 2) {
        // Make B upper triangular
        T cl1, sl1;
        rotg(B(j0 + n2, j0 + n2), B(j1 + n2, j0 + n2), cl1, sl1);
        B(j1 + n2, j0 + n2) = (T)0;
        {
            auto b1 = slice(B, j0 + n2, range(j1 + n2, j2 + n2));
            auto b2 = slice(B, j1 + n2, range(j1 + n2, j2 + n2));
            rot(b1, b2, cl1, sl1);
        }
        // Standard form
        T ssmin, ssmax, cl2, sl2, cr, sr;
        svd22(B(j0 + n2, j0 + n2), B(j0 + n2, j1 + n2), B(j1 + n2, j1 + n2),
              ssmin, ssmax, cl2, sl2, cr, sr);
        if (ssmax < (T)0) {
            cr = -cr;
            sr = -sr;
            ssmin = -ssmin;
            ssmax = -ssmax;
        }
        B(j0 + n2, j0 + n2) = ssmax;
        B(j1 + n2, j1 + n2) = ssmin;
        B(j0 + n2, j1 + n2) = (T)0;
        // Fuse left rotations
        T cl, sl;
        cl = cl1 * cl2 - sl1 * sl2;
        sl = cl2 * sl1 + sl2 * cl1;
        // Apply left rotation
        {
            auto a1 = slice(A, j0 + n2, range(j0 + n2, n));
            auto a2 = slice(A, j1 + n2, range(j0 + n2, n));
            rot(a1, a2, cl, sl);
            auto b1 = slice(B, j0 + n2, range(j2 + n2, n));
            auto b2 = slice(B, j1 + n2, range(j2 + n2, n));
            rot(b1, b2, cl, sl);
            auto q0 = col(Q, j0 + n2);
            auto q1 = col(Q, j1 + n2);
            rot(q0, q1, cl, sl);
        }
        // Apply right rotation
        {
            auto a1 = slice(A, range(0, j2 + n2), j0 + n2);
            auto a2 = slice(A, range(0, j2 + n2), j1 + n2);
            rot(a1, a2, cr, sr);
            auto b1 = slice(B, range(0, j0 + n2), j0 + n2);
            auto b2 = slice(B, range(0, j0 + n2), j1 + n2);
            rot(b1, b2, cr, sr);
            auto z0 = col(Z, j0 + n2);
            auto z1 = col(Z, j1 + n2);
            rot(z0, z1, cr, sr);
        }
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
