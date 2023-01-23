/// @file trmv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_TRMV_HH
#define TLAPACK_LEGACY_TRMV_HH

#include "tlapack/blas/trmv.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {

/**
 * Triangular matrix-vector multiply:
 * \[
 *     x = op(A) x,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * x is a vector,
 * and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero.
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $x = A   x$,
 *     - Op::Trans:     $x = A^T x$,
 *     - Op::ConjTrans: $x = A^H x$.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *                      The diagonal elements of A are not referenced.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] n
 *     Number of rows and columns of the matrix A. n >= 0.
 *
 * @param[in] A
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *
 * @param[in] lda
 *     Leading dimension of A. lda >= max(1, n).
 *
 * @param[in,out] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @ingroup legacy_blas
 */
template <typename TA, typename TX>
void trmv(Layout layout,
          Uplo uplo,
          Op trans,
          Diag diag,
          idx_t n,
          TA const* A,
          idx_t lda,
          TX* x,
          int_t incx)
{
    using internal::colmajor_matrix;

    // check arguments
    tlapack_check_false(layout != Layout::ColMajor &&
                        layout != Layout::RowMajor);
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);
    tlapack_check_false(n < 0);
    tlapack_check_false(lda < n);
    tlapack_check_false(incx == 0);

    // quick return
    if (n == 0) return;

    // for row major, swap lower <=> upper and
    // A => A^T; A^T => A; A^H => conj(A)
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans)
                    ? Op::Trans
                    : ((trans == Op::Trans) ? Op::NoTrans : Op::Conj);
    }

    // Matrix views
    const auto A_ = colmajor_matrix<TA>((TA*)A, n, n, lda);

    tlapack_expr_with_vector(x_, TX, n, x, incx,
                             return trmv(uplo, trans, diag, A_, x_));
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_TRMV_HH
