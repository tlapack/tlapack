/// @file legacy_api/blas/symv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_SYMV_HH
#define TLAPACK_LEGACY_SYMV_HH

#include "tlapack/blas/symv.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {
namespace legacy {

    /**
     * Symmetric matrix-vector multiply:
     * \[
     *     y = \alpha A x + \beta y,
     * \]
     * where alpha and beta are scalars, x and y are vectors,
     * and A is an n-by-n symmetric matrix.
     *
     * Generic implementation for arbitrary data types.
     *
     * @param[in] layout
     *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
     *
     * @param[in] uplo
     *     What part of the matrix A is referenced,
     *     the opposite triangle being assumed from symmetry.
     *     - Uplo::Lower: only the lower triangular part of A is referenced.
     *     - Uplo::Upper: only the upper triangular part of A is referenced.
     *
     * @param[in] n
     *     Number of rows and columns of the matrix A. n >= 0.
     *
     * @param[in] alpha
     *     Scalar alpha. If alpha is zero, A and x are not accessed.
     *
     * @param[in] A
     *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor:
     * n-by-lda].
     *
     * @param[in] lda
     *     Leading dimension of A. lda >= max(1, n).
     *
     * @param[in] x
     *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
     *
     * @param[in] incx
     *     Stride between elements of x. incx must not be zero.
     *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
     *
     * @param[in] beta
     *     Scalar beta. If beta is zero, y need not be set on input.
     *
     * @param[in,out] y
     *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
     *
     * @param[in] incy
     *     Stride between elements of y. incy must not be zero.
     *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
     *
     * @ingroup legacy_blas
     */
    template <typename TA, typename TX, typename TY>
    void symv(Layout layout,
              Uplo uplo,
              idx_t n,
              scalar_type<TA, TX, TY> alpha,
              TA const* A,
              idx_t lda,
              TX const* x,
              int_t incx,
              scalar_type<TA, TX, TY> beta,
              TY* y,
              int_t incy)
    {
        using scalar_t = scalar_type<TA, TX, TY>;

        // check arguments
        tlapack_check_false(layout != Layout::ColMajor &&
                            layout != Layout::RowMajor);
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
        tlapack_check_false(n < 0);
        tlapack_check_false(lda < n);
        tlapack_check_false(incx == 0);
        tlapack_check_false(incy == 0);

        // quick return
        if (n == 0 || ((alpha == scalar_t(0)) && (beta == scalar_t(1)))) return;

        // for row major, swap lower <=> upper
        if (layout == Layout::RowMajor) {
            uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        }

        // Views
        const auto A_ = create_matrix<TA>((TA*)A, n, n, lda);

        if (alpha == scalar_t(0)) {
            tlapack_expr_with_vector(
                y_, TY, n, y, incy,
                if (beta == scalar_t(0)) for (idx_t i = 0; i < n; ++i) y_[i] =
                    TY(0);
                else for (idx_t i = 0; i < n; ++i) y_[i] *= beta);
        }
        else {
            tlapack_expr_with_2vectors(
                x_, TX, n, x, incx, y_, TY, n, y, incy,
                if (beta == scalar_t(0)) return symv(uplo, alpha, A_, x_, y_);
                else return symv(uplo, alpha, A_, x_, beta, y_));
        }
    }

}  // namespace legacy
}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_SYMV_HH
