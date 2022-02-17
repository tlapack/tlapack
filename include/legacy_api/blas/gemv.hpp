// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_GEMV_HH
#define TBLAS_LEGACY_GEMV_HH

#include "blas/utils.hpp"
#include "blas/gemv.hpp"

namespace blas {

/**
 * General matrix-vector multiply:
 * \[
 *     y = \alpha op(A) x + \beta y,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * alpha and beta are scalars, x and y are vectors,
 * and A is an m-by-n matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $y = \alpha A   x + \beta y$,
 *     - Op::Trans:     $y = \alpha A^T x + \beta y$,
 *     - Op::ConjTrans: $y = \alpha A^H x + \beta y$.
 *
 * @param[in] m
 *     Number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *     Number of columns of the matrix A. n >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, A and x are not accessed.
 *
 * @param[in] A
 *     The m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
 *
 * @param[in] lda
 *     Leading dimension of A. lda >= max(1, m) [RowMajor: lda >= max(1, n)].
 *
 * @param[in] x
 *     - If trans = NoTrans:
 *       the n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *     - Otherwise:
 *       the m-element vector x, in an array of length (m-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[in] beta
 *     Scalar beta. If beta is zero, y need not be set on input.
 *
 * @param[in, out] y
 *     - If trans = NoTrans:
 *       the m-element vector y, in an array of length (m-1)*abs(incy) + 1.
 *     - Otherwise:
 *       the n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup gemv
 */
template< typename TA, typename TX, typename TY >
void gemv(
    blas::Layout layout,
    blas::Op trans,
    blas::idx_t m, blas::idx_t n,
    blas::scalar_type<TA, TX, TY> alpha,
    TA const *A, blas::idx_t lda,
    TX const *x, blas::int_t incx,
    blas::scalar_type<TA, TX, TY> beta,
    TY *y, blas::int_t incy )
{
    typedef blas::scalar_type<TA, TX, TY> scalar_t;
    using blas::internal::colmajor_matrix;
    
    // constants
    const scalar_t zero( 0.0 );
    const scalar_t one( 1.0 );

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( lda < ((layout == Layout::ColMajor) ? m : n) );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // quick return
    if (m == 0 || n == 0 || (alpha == zero && beta == one))
        return;

    // Transpose if Row Major
    if (layout == Layout::RowMajor) {
        // A => A^T; A^T => A; A^H => conj(A)
        std::swap( m, n );
        trans = (trans == Op::NoTrans)
              ? Op::Trans
              : ((trans == Op::Trans) ? Op::NoTrans : Op::Conj);
    }

    // Initialize indexes
    blas::idx_t lenx = ((trans == Op::NoTrans || trans == Op::Conj) ? n : m);
    blas::idx_t leny = ((trans == Op::NoTrans || trans == Op::Conj) ? m : n);
    
    // Matrix views
    const auto _A = colmajor_matrix<TA>( (TA*)A, m, n, lda );

    tlapack_expr_with_2vectors(
        _x, TX, lenx, x, incx,
        _y, TY, leny, y, incy,
        gemv( trans, alpha, _A, _x, beta, _y )
    );
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_GEMV_HH
