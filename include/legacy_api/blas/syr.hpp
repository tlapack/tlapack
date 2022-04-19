// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_SYR_HH
#define TBLAS_LEGACY_SYR_HH

#include "blas/utils.hpp"
#include "blas/syr.hpp"

namespace blas {

/**
 * Symmetric matrix rank-1 update:
 * \[
 *     A = \alpha x x^T + A,
 * \]
 * where alpha is a scalar, x is a vector,
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
 *     Scalar alpha. If alpha is zero, A is not updated.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[in,out] A
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *
 * @param[in] lda
 *     Leading dimension of A. lda >= max(1, n).
 *
 * @ingroup syr
 */
template< typename TA, typename TX >
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::idx_t n,
    blas::scalar_type<TA, TX> alpha,
    TX const *x, blas::int_t incx,
    TA       *A, blas::idx_t lda )
{
    using blas::internal::colmajor_matrix;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( lda < n );

    // quick return
    if (n == 0)
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }
    
    // Matrix views
    auto _A = colmajor_matrix<TA>( A, n, n, lda );

    tlapack_expr_with_vector( _x, TX, n, x, incx, syr( uplo, alpha, _x, _A ) );
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_SYR_HH
