// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_GEMV_HH
#define TLAPACK_LEGACY_GEMV_HH

#include "tlapack/legacy_api/base/utils.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/blas/gemv.hpp"

namespace tlapack {

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
 * @param[in,out] y
 *     - If trans = NoTrans:
 *       the m-element vector y, in an array of length (m-1)*abs(incy) + 1.
 *     - Otherwise:
 *       the n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup legacy_blas
 */
template< typename TA, typename TX, typename TY >
void gemv(
    Layout layout,
    Op trans,
    idx_t m, idx_t n,
    scalar_type<TA, TX, TY> alpha,
    TA const *A, idx_t lda,
    TX const *x, int_t incx,
    scalar_type<TA, TX, TY> beta,
    TY *y, int_t incy )
{
    using internal::colmajor_matrix;
    using scalar_t = scalar_type<TA, TX, TY>;

    // check arguments
    tlapack_check_false( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    tlapack_check_false( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    tlapack_check_false( m < 0 );
    tlapack_check_false( n < 0 );
    tlapack_check_false( lda < ((layout == Layout::ColMajor) ? m : n) );
    tlapack_check_false( incx == 0 );
    tlapack_check_false( incy == 0 );

    // quick return
    if ( m == 0 || n == 0 ||
        ((alpha == scalar_t(0)) && (beta == scalar_t(1))) ) return;

    // Transpose if Row Major
    if (layout == Layout::RowMajor) {
        // A => A^T; A^T => A; A^H => conj(A)
        std::swap( m, n );
        trans = (trans == Op::NoTrans)
              ? Op::Trans
              : ((trans == Op::Trans) ? Op::NoTrans : Op::Conj);
    }

    // Initialize indexes
    idx_t lenx = ((trans == Op::NoTrans || trans == Op::Conj) ? n : m);
    idx_t leny = ((trans == Op::NoTrans || trans == Op::Conj) ? m : n);
    
    // Matrix views
    const auto A_ = colmajor_matrix<TA>( (TA*)A, m, n, lda );

    if( alpha == scalar_t(0) ) {
        tlapack_expr_with_vector( y_, TY, leny, y, incy,
            if( beta == scalar_t(0) )
                for(idx_t i = 0; i < leny; ++i)
                    y_[i] = TY(0);
            else
                for(idx_t i = 0; i < leny; ++i)
                    y_[i] *= beta
        );
    }
    else {
        tlapack_expr_with_2vectors(
            x_, TX, lenx, x, incx,
            y_, TY, leny, y, incy,
            if( beta == scalar_t(0) )
                return gemv( trans, alpha, A_, x_, y_ );
            else
                return gemv( trans, alpha, A_, x_, beta, y_ )
        );
    }
}

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_LEGACY_GEMV_HH
