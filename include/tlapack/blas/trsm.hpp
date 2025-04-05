/// @file trsm.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_TRSM_HH
#define TLAPACK_BLAS_TRSM_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Solve the triangular matrix-vector equation
 * \[
 *     op(A) X = \alpha B,
 * \]
 * or
 * \[
 *     X op(A) = \alpha B,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * X and B are m-by-n matrices, and A is an m-by-m or n-by-n, unit or non-unit,
 * upper or lower triangular matrix.
 *
 * No test for singularity or near-singularity is included in this
 * routine. Such tests must be performed before calling this routine.
 *
 * @param[in] side
 *     Whether $op(A)$ is on the left or right of X:
 *     - Side::Left:  $op(A) X = B$.
 *     - Side::Right: $X op(A) = B$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero:
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
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
 * @param[in,out] B
 *      On entry, the m-by-n matrix B.
 *      On exit,  the m-by-n matrix X.
 *
 * @ingroup blas3
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<matrixB_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<alpha_t, T> > = 0>
void trsm(Side side,
          Uplo uplo,
          Op trans,
          Diag diag,
          const alpha_t& alpha,
          const matrixA_t& A,
          matrixB_t& B)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using TB = type_t<matrixB_t>;

    // constants
    const idx_t m = nrows(B);
    const idx_t n = ncols(B);

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);
    tlapack_check_false(nrows(A) != ncols(A));
    tlapack_check_false(nrows(A) != ((side == Side::Left) ? m : n));

    if (side == Side::Left) {
        using scalar_t = scalar_type<alpha_t, TB>;
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i)
                        B(i, j) *= alpha;
                    for (idx_t k = m - 1; k != idx_t(-1); --k) {
                        if (diag == Diag::NonUnit) B(k, j) /= A(k, k);
                        for (idx_t i = 0; i < k; ++i)
                            B(i, j) -= A(i, k) * B(k, j);
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i)
                        B(i, j) *= alpha;
                    for (idx_t k = 0; k < m; ++k) {
                        if (diag == Diag::NonUnit) B(k, j) /= A(k, k);
                        for (idx_t i = k + 1; i < m; ++i)
                            B(i, j) -= A(i, k) * B(k, j);
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i) {
                        scalar_t sum = alpha * B(i, j);
                        for (idx_t k = 0; k < i; ++k)
                            sum -= A(k, i) * B(k, j);
                        B(i, j) = (diag == Diag::NonUnit) ? sum / A(i, i) : sum;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = m - 1; i != idx_t(-1); --i) {
                        scalar_t sum = alpha * B(i, j);
                        for (idx_t k = i + 1; k < m; ++k)
                            sum -= A(k, i) * B(k, j);
                        B(i, j) = (diag == Diag::NonUnit) ? sum / A(i, i) : sum;
                    }
                }
            }
        }
        else {  // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i) {
                        scalar_t sum = alpha * B(i, j);
                        for (idx_t k = 0; k < i; ++k)
                            sum -= conj(A(k, i)) * B(k, j);
                        B(i, j) =
                            (diag == Diag::NonUnit) ? sum / conj(A(i, i)) : sum;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = m - 1; i != idx_t(-1); --i) {
                        scalar_t sum = alpha * B(i, j);
                        for (idx_t k = i + 1; k < m; ++k)
                            sum -= conj(A(k, i)) * B(k, j);
                        B(i, j) =
                            (diag == Diag::NonUnit) ? sum / conj(A(i, i)) : sum;
                    }
                }
            }
        }
    }
    else {  // side == Side::Right
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i)
                        B(i, j) *= alpha;
                    for (idx_t k = 0; k < j; ++k) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) -= B(i, k) * A(k, j);
                    }
                    if (diag == Diag::NonUnit) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) /= A(j, j);
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = n - 1; j != idx_t(-1); --j) {
                    for (idx_t i = 0; i < m; ++i)
                        B(i, j) *= alpha;
                    for (idx_t k = j + 1; k < n; ++k) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) -= B(i, k) * A(k, j);
                    }
                    if (diag == Diag::NonUnit) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) /= A(j, j);
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for (idx_t k = n - 1; k != idx_t(-1); --k) {
                    if (diag == Diag::NonUnit) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) /= A(k, k);
                    }
                    for (idx_t j = 0; j < k; ++j) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) -= B(i, k) * A(j, k);
                    }
                    for (idx_t i = 0; i < m; ++i)
                        B(i, k) *= alpha;
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t k = 0; k < n; ++k) {
                    if (diag == Diag::NonUnit) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) /= A(k, k);
                    }
                    for (idx_t j = k + 1; j < n; ++j) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) -= B(i, k) * A(j, k);
                    }
                    for (idx_t i = 0; i < m; ++i)
                        B(i, k) *= alpha;
                }
            }
        }
        else {  // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for (idx_t k = n - 1; k != idx_t(-1); --k) {
                    if (diag == Diag::NonUnit) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) /= conj(A(k, k));
                    }
                    for (idx_t j = 0; j < k; ++j) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) -= B(i, k) * conj(A(j, k));
                    }
                    for (idx_t i = 0; i < m; ++i)
                        B(i, k) *= alpha;
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t k = 0; k < n; ++k) {
                    if (diag == Diag::NonUnit) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) /= conj(A(k, k));
                    }
                    for (idx_t j = k + 1; j < n; ++j) {
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) -= B(i, k) * conj(A(j, k));
                    }
                    for (idx_t i = 0; i < m; ++i)
                        B(i, k) *= alpha;
                }
            }
        }
    }
}

#ifdef TLAPACK_USE_LAPACKPP

template <TLAPACK_LEGACY_MATRIX matrixA_t,
          TLAPACK_LEGACY_MATRIX matrixB_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<matrixB_t>,
          enable_if_allow_optblas_t<pair<matrixA_t, T>,
                                    pair<matrixB_t, T>,
                                    pair<alpha_t, T> > = 0>
void trsm(Side side,
          Uplo uplo,
          Op trans,
          Diag diag,
          const alpha_t alpha,
          const matrixA_t& A,
          matrixB_t& B)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto B_ = legacy_matrix(B);

    // Constants to forward
    constexpr Layout L = layout<matrixB_t>;
    const auto& m = B_.m;
    const auto& n = B_.n;

    // Warnings for NaNs and Infs
    if (alpha == alpha_t(0))
        tlapack_warning(
            -5, "Infs and NaNs in A or B will not propagate to B on output");

    return ::blas::trsm((::blas::Layout)L, (::blas::Side)side,
                        (::blas::Uplo)uplo, (::blas::Op)trans,
                        (::blas::Diag)diag, m, n, alpha, A_.ptr, A_.ldim,
                        B_.ptr, B_.ldim);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_TRSM_HH
