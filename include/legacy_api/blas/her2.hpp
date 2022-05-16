// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_HER2_HH__
#define __TLAPACK_LEGACY_HER2_HH__

#include "legacy_api/base/utils.hpp"
#include "legacy_api/base/types.hpp"
#include "blas/her2.hpp"

namespace tlapack {

/**
 * Hermitian matrix rank-2 update:
 * \[
 *     A = \alpha x y^H + \text{conj}(\alpha) y x^H + A,
 * \]
 * where alpha is a scalar, x and y are vectors,
 * and A is an n-by-n Hermitian matrix.
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
 *     Scalar alpha. If alpha is zero, A is not updated.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[in] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @param[in,out] A
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 *
 * @param[in] lda
 *     Leading dimension of A. lda >= max(1, n).
 *
 * @ingroup her2
 */
template< typename TA, typename TX, typename TY >
void her2(
    Layout layout,
    Uplo  uplo,
    idx_t n,
    scalar_type<TA, TX, TY> alpha,
    TX const *x, int_t incx,
    TY const *y, int_t incy,
    TA *A, idx_t lda )
{
    using internal::colmajor_matrix;

    // check arguments
    tblas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    tblas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    tblas_error_if( n < 0 );
    tblas_error_if( incx == 0 );
    tblas_error_if( incy == 0 );
    tblas_error_if( lda < n );

    // quick return
    if (n == 0)
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }
    
    // Matrix views
    auto A_ = colmajor_matrix<TA>( A, n, n, lda );

    tlapack_expr_with_2vectors(
        _x, TX, n, x, incx,
        _y, TY, n, y, incy,
        return her2( uplo, alpha, _x, _y, A_ )
    );
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_LEGACY_HER2_HH__
