/// @file trevc3.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dtrevc.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC_HH
#define TLAPACK_TREVC_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/lapack/trevc3_backsolve.hpp"
#include "tlapack/lapack/trevc3_forwardsolve.hpp"

namespace tlapack {

/**
 * Options struct for multishift_qr().
 */
struct Trevc3Opts {
    // Size of the eigenvector blocks
    size_t block_size = 64;
    // Blocksize used in the backward and forward substitution
    size_t block_size_solve = 64;
};

enum class HowMny : char {
    All = 'A',    ///< all eigenvectors
    Back = 'B',   ///< all eigenvectors, backtransformed by input matrix
    Select = 'S'  ///< selected eigenvectors
};

/** Worspace query of TREVC3()
 *
 * @param[in] side tlapack::Side
 * @param[in] howmny tlapack::HowMny
 * @param[in,out] select Boolean array of size n
 * @param[in] T      n-by-n upper quasi-triangular matrix.
 * @param[in,out] Vl    n-by-m matrix
 * @param[in,out] Vr    n-by-m matrix
 * @param[in] opts Trevc3Opts
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_VECTOR select_t,
          TLAPACK_MATRIX matrix_T_t,
          TLAPACK_MATRIX matrix_Vl_t,
          TLAPACK_MATRIX matrix_Vr_t>
WorkInfo trevc3_worksize(const side_t side,
                         const HowMny howmny,
                         select_t& select,
                         const matrix_T_t& T,
                         matrix_Vl_t& Vl,
                         matrix_Vr_t& Vr,
                         const Trevc3Opts& opts = {})
{
    using idx_t = size_type<matrix_T_t>;

    const idx_t n = ncols(T);

    // Space for single and double back/forward substitution
    WorkInfo workinfo(n * 3);

    // Space for blocked back/forward substitution
    workinfo += WorkInfo(opts.block_size_solve + 1);

    // Space to temporarily store eigenvector block
    workinfo += WorkInfo(n, opts.block_size + 1);

    // Space to store the backtransformed eigenvector block
    if (howmny == HowMny::Back) {
        workinfo += WorkInfo(n, opts.block_size + 1);
    }

    return workinfo;
}

/**
 *
 * TREVC3 computes some or all of the right and/or left eigenvectors of
 * an upper quasi-triangular matrix T.
 * Matrices of this type are produced by the Schur factorization of
 * a general matrix:  A = Q*T*Q**T
 *
 * The right eigenvector x and the left eigenvector y of T corresponding
 * to an eigenvalue w are defined by:
 *
 *    T*x = w*x,     (y**T)*T = w*(y**T)
 *
 * where y**T denotes the transpose of the vector y.
 * The eigenvalues are not input to this routine, but are read directly
 * from the diagonal blocks of T.
 *
 * This routine returns the matrices X and/or Y of right and left
 * eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
 * input matrix. If Q is the orthogonal factor that reduces a matrix
 * A to Schur form T, then Q*X and Q*Y are the matrices of right and
 * left eigenvectors of A.
 *
 * This is a level-3 BLAS version of trevc.
 *
 * @param[in] side tlapack::Side
 *                 Specifies whether right or left eigenvectors are required:
 *                 = Side::Right: right eigenvectors only;
 *                 = Side::Left: left eigenvectors only;
 *                 = Side::Both: both right and left eigenvectors.
 *
 * @param[in] howmny tlapack::HowMny
 *                   Specifies how many eigenvectors are to be computed:
 *                   = HowMny::All: all right and/or left eigenvectors
 *                   = HowMny::Back: all right and/or left eigenvectors,
 *                   backtransformed by input matrix
 *                   = HowMny::Select: selected right and/or left eigenvectors
 *                   as indicated by the boolean array select.
 *
 * @param[in,out] select Boolean array of size n,
 *                   where n is the order of the matrix
 *                   T. On input, select specifies which eigenvectors are to be
 *                   computed. On output, select indicates which eigenvectors
 *                   were computed. Not referenced if howmny is not
 *                   HowMny::Select.
 *
 * @param[in] T      n-by-n upper quasi-triangular matrix.
 *                   The matrix T is in Schur canonical form
 *
 * @param[out] Vl    n-by-m matrix, where m is the number of left eigenvectors
 *                   to be computed, as specified by howmny (or n if howmny !=
 *                   HowMny::Select)
 *                   On entry, if howmny == HowMny::Back, Vl must contain an
 *                   n-by-n matrix Q (usually the orthogonal matrix
 *                   that reduces A to Schur form).
 *                   On exit, Vl contains:
 *                   HowMny::All: the matrix Y of the left eigenvectors of T
 *                   HowMny::Back: the matrix Q*Y
 *                   HowMny::Select: the left eigenvectors of T specified by
 *                   the boolean array select, stored consecutively in the
 *                   columns of Vl, in the same order as their eigenvalues.
 *
 * @param[out] Vr    n-by-m matrix, where m is the number of right eigenvectors
 *                   to be computed, as specified by howmny (or n if howmny !=
 *                   HowMny::Select)
 *                   On entry, if howmny == HowMny::Back, Vr must contain an
 *                   n-by-n matrix Q (usually the orthogonal matrix
 *                   that reduces A to Schur form).
 *                   On exit, Vr contains:
 *                   HowMny::All: the matrix X of the right eigenvectors of T
 *                   HowMny::Back: the matrix Q*X
 *                   HowMny::Select: the right eigenvectors of T specified by
 *                   the boolean array select, stored consecutively in the
 *                   columns of Vr, in the same order as their eigenvalues.
 *
 * @param[out] work  Workspace vector, size specified by the workspace query
 *
 * @param[in] opts Trevc3Opts
 *
 * @ingroup trevc3
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_VECTOR select_t,
          TLAPACK_MATRIX matrix_T_t,
          TLAPACK_MATRIX matrix_Vl_t,
          TLAPACK_MATRIX matrix_Vr_t,
          TLAPACK_WORKSPACE work_t>
int trevc3_work(const side_t side,
                const HowMny howmny,
                select_t& select,
                const matrix_T_t& T,
                matrix_Vl_t& Vl,
                matrix_Vr_t& Vr,
                work_t& work,
                const Trevc3Opts& opts = {})
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<TT>;

    const idx_t n = nrows(T);
    // Number of columns of Vl and Vr
    // const idx_t mm = max(ncols(Vl), ncols(Vr));
    idx_t mm;
    if (side == Side::Both) {
        mm = std::min<idx_t>(ncols(Vl), ncols(Vr));
    }
    else if (side == Side::Left) {
        mm = ncols(Vl);
    }
    else {
        mm = ncols(Vr);
    }

    // Quick return
    if (n == 0) return 0;

    idx_t m = 0;  // Actual number of eigenvectors to compute
    if (howmny == HowMny::Select) {
        // Set m to the number of columns required to store the selected
        // eigenvectors.
        // If necessary, the array select is standardized for complex
        // conjugate pairs so that select[j] is true and select[j+1] is false.
        idx_t j = 0;
        while (j < n) {
            bool pair = false;
            if (j < n - 1) {
                if (T(j + 1, j) != TT(0)) {
                    pair = true;
                }
            }
            if (!pair) {
                if (select[j]) {
                    m++;
                }
                j++;
            }
            else {
                if (select[j] || select[j + 1]) {
                    select[j] = true;
                    select[j + 1] = true;
                    m += 2;
                }
                j += 2;
            }
        }
    }
    else {
        m = n;
    }

    // Make sure that the matrices Vl and Vr have enough space
    tlapack_check(mm >= m);

    const idx_t nb = opts.block_size;  // Block size

    auto [X, work2] = reshape(work, n, opts.block_size + 1);

    if (side == Side::Right || side == Side::Both) {
        //
        // Compute right eigenvectors.
        //
        idx_t iVr = m - 1;  // current column of Vr to store the eigenvector
        for (idx_t ii = 0; ii < n;) {
            idx_t i = n - 1 - ii;
            // First eigenvector to compute within this block
            idx_t i_start;
            // Last eigenvector to compute within this block (inclusive)
            idx_t i_end;

            if (howmny == HowMny::Select) {
                // Find the first eigenvalue with index <= i with select ==
                // true
                for (idx_t jj = 0; jj < i + 1; jj++) {
                    idx_t j = i - jj;
                    if (select[j]) {
                        i_end = j;
                        break;
                    }
                }
                // Find the last eigenvalue with index <= i_end with select ==
                // true
                i_start = 0;
                for (idx_t jj = 0; jj < i_end + 1; jj++) {
                    idx_t j = i_end - jj;
                    if (jj >= nb) {
                        // Don't make the block too big
                        i_start = j + 1;
                        break;
                    }
                    bool pair = false;
                    if (!select[j]) {
                        i_start = j + 1;
                        break;
                    }
                }
            }
            else {
                i_end = i;
                i_start = (i >= nb) ? i - nb + 1 : 0;
            }

            // Make sure we don't split 2x2 blocks
            if (i_start > 0) {
                if (T(i_start, i_start - 1) != TT(0)) {
                    i_start--;
                }
            }

            // The true block size
            idx_t nb2 = i_end - i_start + 1;

            // Calculate the current block of eigenvectors of T
            trevc3_backsolve(T, X, work2, i_start, i_end + 1,
                             opts.block_size_solve);

            // If required, backtransform the eigenvectors to the original
            // matrix, otherwise copy them to Vr
            if (howmny == HowMny::Back) {
                auto Q_slice = slice(Vr, range(0, n), range(0, i_end + 1));
                auto X_block = slice(X, range(0, i_end + 1), range(0, nb2));
                auto [Qx, work3] = reshape(work2, n, nb2);

                gemm(Op::NoTrans, Op::NoTrans, TT(1), Q_slice, X_block, TT(0),
                     Qx);

                auto Vr_block =
                    slice(Vr, range(0, n), range(i_start, i_end + 1));
                lacpy(Uplo::General, Qx, Vr_block);
            }
            else {
                auto Vr_block =
                    slice(Vr, range(0, n), range(iVr - nb2 + 1, iVr + 1));
                auto X_block = slice(X, range(0, n), range(0, nb2));
                lacpy(Uplo::General, X_block, Vr_block);
                iVr -= nb2;
            }

            ii += nb2;
        }
    }

    if (side == Side::Left || side == Side::Both) {
        //
        // Compute left eigenvectors.
        //
        idx_t iVl = 0;
        for (idx_t i = 0; i < n;) {
            // First eigenvector to compute within this block
            idx_t i_start;
            // Last eigenvector to compute within this block (inclusive)
            idx_t i_end;

            if (howmny == HowMny::Select) {
                // Find the first eigenvalue with index <= i with select ==
                // true
                for (idx_t j = i; j < n; j++) {
                    if (select[j]) {
                        i_start = j;
                        break;
                    }
                }
                // Find the last eigenvalue with index <= i_end with select ==
                // true
                i_end = n - 1;
                for (idx_t j = i_start + 1; j < n; j++) {
                    if (j > i_start + nb) {
                        // Don't make the block too big
                        i_end = j - 1;
                        break;
                    }
                    bool pair = false;
                    if (!select[j]) {
                        i_end = j - 1;
                        break;
                    }
                }
            }
            else {
                i_start = i;
                i_end = min<idx_t>(n - 1, i + nb - 1);
            }

            // Make sure we don't split 2x2 blocks
            if (i_end + 1 < n) {
                if (T(i_end + 1, i_end) != TT(0)) {
                    i_end++;
                }
            }

            // The true block size
            idx_t nb2 = i_end - i_start + 1;

            // Calculate the current block of eigenvectors of T
            trevc3_forwardsolve(T, X, work2, i_start, i_end + 1,
                                opts.block_size_solve);

            // If required, backtransform the eigenvectors to the original
            // matrix, otherwise copy them to Vl
            if (howmny == HowMny::Back) {
                auto Q_slice = slice(Vl, range(0, n), range(i_start, n));
                auto X_block = slice(X, range(i_start, n), range(0, nb2));
                auto [Qx, work3] = reshape(work2, n, nb2);

                gemm(Op::NoTrans, Op::NoTrans, TT(1), Q_slice, X_block, TT(0),
                     Qx);

                auto Vl_block =
                    slice(Vl, range(0, n), range(i_start, i_end + 1));
                lacpy(Uplo::General, Qx, Vl_block);
            }
            else {
                auto Vl_block = slice(Vl, range(0, n), range(iVl, iVl + nb2));
                auto X_block = slice(X, range(0, n), range(0, nb2));
                lacpy(Uplo::General, X_block, Vl_block);
                iVl += nb2;
            }

            i += nb2;
        }
    }

    return 0;
}

/**
 *
 * TREVC3 computes some or all of the right and/or left eigenvectors of
 * an upper quasi-triangular matrix T.
 * Matrices of this type are produced by the Schur factorization of
 * a general matrix:  A = Q*T*Q**T
 *
 * The right eigenvector x and the left eigenvector y of T corresponding
 * to an eigenvalue w are defined by:
 *
 *    T*x = w*x,     (y**T)*T = w*(y**T)
 *
 * where y**T denotes the transpose of the vector y.
 * The eigenvalues are not input to this routine, but are read directly
 * from the diagonal blocks of T.
 *
 * This routine returns the matrices X and/or Y of right and left
 * eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
 * input matrix. If Q is the orthogonal factor that reduces a matrix
 * A to Schur form T, then Q*X and Q*Y are the matrices of right and
 * left eigenvectors of A.
 *
 * This is a level-3 BLAS version of trevc.
 *
 * @param[in] side tlapack::Side
 *                 Specifies whether right or left eigenvectors are required:
 *                 = Side::Right: right eigenvectors only;
 *                 = Side::Left: left eigenvectors only;
 *                 = Side::Both: both right and left eigenvectors.
 *
 * @param[in] howmny tlapack::HowMny
 *                   Specifies how many eigenvectors are to be computed:
 *                   = HowMny::All: all right and/or left eigenvectors
 *                   = HowMny::Back: all right and/or left eigenvectors,
 *                   backtransformed by input matrix
 *                   = HowMny::Select: selected right and/or left eigenvectors
 *                   as indicated by the boolean array select.
 *
 * @param[in,out] select Boolean array of size n,
 *                   where n is the order of the matrix
 *                   T. On input, select specifies which eigenvectors are to be
 *                   computed. On output, select indicates which eigenvectors
 *                   were computed. Not referenced if howmny is not
 *                   HowMny::Select.
 *
 * @param[in] T      n-by-n upper quasi-triangular matrix.
 *                   The matrix T is in Schur canonical form
 *
 * @param[out] Vl    n-by-m matrix, where m is the number of left eigenvectors
 *                   to be computed, as specified by howmny (or n if howmny !=
 *                   HowMny::Select)
 *                   On entry, if howmny == HowMny::Back, Vl must contain an
 *                   n-by-n matrix Q (usually the orthogonal matrix
 *                   that reduces A to Schur form).
 *                   On exit, Vl contains:
 *                   HowMny::All: the matrix Y of the left eigenvectors of T
 *                   HowMny::Back: the matrix Q*Y
 *                   HowMny::Select: the left eigenvectors of T specified by
 *                   the boolean array select, stored consecutively in the
 *                   columns of Vl, in the same order as their eigenvalues.
 *
 * @param[out] Vr    n-by-m matrix, where m is the number of right eigenvectors
 *                   to be computed, as specified by howmny (or n if howmny !=
 *                   HowMny::Select)
 *                   On entry, if howmny == HowMny::Back, Vr must contain an
 *                   n-by-n matrix Q (usually the orthogonal matrix
 *                   that reduces A to Schur form).
 *                   On exit, Vr contains:
 *                   HowMny::All: the matrix X of the right eigenvectors of T
 *                   HowMny::Back: the matrix Q*X
 *                   HowMny::Select: the right eigenvectors of T specified by
 *                   the boolean array select, stored consecutively in the
 *                   columns of Vr, in the same order as their eigenvalues.
 *
 * @param[in] opts Trevc3Opts
 *
 * @ingroup trevc3
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_VECTOR select_t,
          TLAPACK_MATRIX matrix_T_t,
          TLAPACK_MATRIX matrix_Vl_t,
          TLAPACK_MATRIX matrix_Vr_t>
int trevc3(const side_t side,
           const HowMny howmny,
           select_t& select,
           const matrix_T_t& T,
           matrix_Vl_t& Vl,
           matrix_Vr_t& Vr,
           const Trevc3Opts& opts = {})
{
    WorkInfo workinfo = trevc3_worksize(side, howmny, select, T, Vl, Vr, opts);

    using TT = type_t<matrix_T_t>;
    Create<vector_type<matrix_T_t>> new_vector;
    std::vector<TT> work_;
    auto work = new_vector(work_, workinfo.m * workinfo.n);
    return trevc3_work(side, howmny, select, T, Vl, Vr, work, opts);
}
}  // namespace tlapack

#endif  // TLAPACK_TREVC_HH
