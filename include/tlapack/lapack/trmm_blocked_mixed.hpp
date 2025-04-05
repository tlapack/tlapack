/// @file trmm_blocked_mixed.hpp
/// @author Weslley S Pereira, National Renewable Energy Laboratory, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRMM_BLOCKED_MIXED_HH
#define TLAPACK_TRMM_BLOCKED_MIXED_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/lacpy.hpp"

namespace tlapack {

/**
 * Options struct for trmm_blocked_mixed
 */
struct TrmmBlockedOpts {
    size_t nb = 32;  ///< Block size
};

/**
 * Triangular matrix-matrix multiply using a blocked algorithm.
 *
 * In iteration i, the algorithm computes $B_{0i} += \alpha A_{0i,i} B_i$ and
 * then $B_i := \alpha A_{i,i} B_i$, where i is the block index. See details in
 * the implementation. $B_i$ is cast to the precision type of `work`, enabling
 * mixed precision.
 *
 * @todo Implement the algorithm for the remaining cases. See what is missing
 * below.
 *
 * @param[in] side
 *     Whether $op(A)$ is on the left or right of B:
 *     - Side::Left:  $B = \alpha op(A) B$.
 *     - Side::Right: $B = \alpha B op(A)$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero:
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *     - Uplo::General is illegal (see gemm() instead).
 *
 * @param[in] trans
 *     The form of $op(A)$:
 *     - Op::NoTrans:   $op(A) = A$.
 *     - Op::Trans:     $op(A) = A^T$.
 *     - Op::ConjTrans: $op(A) = A^H$.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] alpha Scalar.
 * @param[in] A
 *     - If side = Left: a m-by-m matrix.
 *     - If side = Right: a n-by-n matrix.
 * @param[in,out] B A m-by-n matrix.
 * @param work nb-by-n workspace that also informs the precision type to case B.
 * @param[in] opts Options.
 *
 * @ingroup blas3
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_UPLO uplo_t,
          TLAPACK_OP op_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_SMATRIX matrixA_t,
          TLAPACK_SMATRIX matrixB_t,
          TLAPACK_WORKSPACE work_t>
void trmm_blocked_mixed(
    side_t side,
    uplo_t uplo,
    op_t trans,
    diag_t diag,
    const scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>& alpha,
    const matrixA_t& A,
    matrixB_t& B,
    work_t& work,
    const TrmmBlockedOpts& opts = {})
{
    {
        // data traits
        using idx_t = size_type<matrixA_t>;
        using range = std::pair<idx_t, idx_t>;

        // constants
        const idx_t m = nrows(B);
        const idx_t n = ncols(B);
        const idx_t nb = min(opts.nb, m);

        // check arguments
        tlapack_check_false(side != Side::Left && side != Side::Right);
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
        tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                            trans != Op::ConjTrans);
        tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);
        tlapack_check_false(nrows(A) != ncols(A));
        tlapack_check_false(nrows(A) != ((side == Side::Left) ? m : n));

        // Matrix W
        auto [W, work1] = reshape(work, nb, n);

        using real_t = real_type<type_t<matrixB_t>>;
        if (side == Side::Left) {
            if (trans == Op::NoTrans) {
                if (uplo == Uplo::Upper) {
                    for (idx_t i = 0; i < m; i += nb) {
                        const idx_t ib = min(nb, m - i);

                        const auto A0i =
                            slice(A, range(0, i), range(i, i + ib));
                        const auto Aii =
                            slice(A, range(i, i + ib), range(i, i + ib));

                        auto B0 = rows(B, range(0, i));
                        auto Bi = rows(B, range(i, i + ib));
                        auto BiLowPrecision = rows(W, range(0, ib));

                        // B0 += alpha * A0i * Bi in mixed precision
                        lacpy(GENERAL, Bi, BiLowPrecision);
                        gemm(NO_TRANS, NO_TRANS, alpha, A0i, BiLowPrecision,
                             real_t(1), B0);

                        // Bi = alpha * Aii * Bi in mixed precision
                        trmm(side, uplo, trans, diag, alpha, Aii, Bi);
                    }
                }
                else {  // uplo == Uplo::Lower
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
            }
            else if (trans == Op::Trans) {
                if (uplo == Uplo::Upper) {
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
                else {  // uplo == Uplo::Lower
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
            }
            else {  // trans == Op::ConjTrans
                if (uplo == Uplo::Upper) {
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
                else {  // uplo == Uplo::Lower
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
            }
        }
        else {  // side == Side::Right
            if (trans == Op::NoTrans) {
                if (uplo == Uplo::Upper) {
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
                else {  // uplo == Uplo::Lower
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
            }
            else if (trans == Op::Trans) {
                if (uplo == Uplo::Upper) {
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
                else {  // uplo == Uplo::Lower
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
            }
            else {  // trans == Op::ConjTrans
                if (uplo == Uplo::Upper) {
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
                else {  // uplo == Uplo::Lower
                    tlapack_error(1, "Blocked version of trsm not implemented");
                }
            }
        }
    }
}

}  // namespace tlapack

#endif
