// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SYMV_HH
#define BLAS_SYMV_HH

#include "blas/utils.hpp"

namespace blas {

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
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
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
 * @param[in, out] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup symv
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t, 
    class alpha_t, class beta_t >
void symv(
    Uplo uplo,
    const alpha_t alpha, const matrixA_t& A, const vectorX_t& x,
    const beta_t& beta, vectorY_t& y )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TX    = type_t< vectorX_t >;
    using idx_t = size_type< matrixA_t >;

    // using
    using scalar_t = scalar_type<alpha_t,TA,TX>;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( ncols(A) == n );
    blas_error_if( size(x)  == n );
    blas_error_if( size(y)  == n );

    // form y = beta*y
    if (beta != beta_t(1)) {
        if (beta == beta_t(0)) {
            for (idx_t i = 0; i < n; ++i)
                y(i) = beta_t(0);
        }
        else {
            for (idx_t i = 0; i < n; ++i)
                y(i) *= beta;
        }
    }

    if (uplo == Uplo::Upper) {
        // A is stored in upper triangle
        // form y += alpha * A * x
            // unit stride
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha*x(j);
            auto tmp2 = scalar_t(0);
            for (idx_t i = 0; i < j; ++i) {
                y(i) += tmp1 * A(i,j);
                tmp2 += A(i,j) * x(i);
            }
            y(j) += tmp1 * A(j,j) + alpha * tmp2;
        }
    }
    else {
        // A is stored in lower triangle
        // form y += alpha * A * x
        for (idx_t j = 0; j < n; ++j) {
            scalar_t tmp1 = alpha*x(j);
            auto tmp2 = scalar_t(0);
            for (idx_t i = j+1; i < n; ++i) {
                y(i) += tmp1 * A(i,j);
                tmp2 += A(i,j) * x(i);
            }
            y(j) += tmp1 * A(j,j) + alpha * tmp2;
        }
    }
}

template< typename TA, typename TX, typename TY >
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::idx_t n,
    blas::scalar_type<TA, TX, TY> alpha,
    TA const *A, blas::idx_t lda,
    TX const *x, blas::int_t incx,
    blas::scalar_type<TA, TX, TY> beta,
    TY *y, blas::int_t incy )
{
    typedef blas::scalar_type<TA, TX, TY> scalar_t;
    using blas::internal::colmajor_matrix;
    using blas::internal::vector;

    // constants
    const scalar_t zero( 0.0 );
    const scalar_t one( 1.0 );

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // quick return
    if (n == 0 || (alpha == zero && beta == one))
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    // Views
    const auto _A = colmajor_matrix<TA>( (TA*)A, n, n, lda );
    const auto _x = vector<TX>(
        (TX*) &x[(incx > 0 ? 0 : (-n + 1)*incx)],
        n, incx );
    auto _y = vector<TY>(
        &y[(incy > 0 ? 0 : (-n + 1)*incy)],
        n, incy );

    if (alpha == zero) {
        // form y = beta*y
        if (beta != one) {
            if (beta == zero) {
                for (idx_t i = 0; i < n; ++i)
                    _y(i) = zero;
            }
            else {
                for (idx_t i = 0; i < n; ++i)
                    _y(i) *= beta;
            }
        }
        return;
    }

    symv( uplo, alpha, _A, _x, beta, _y );
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYMV_HH
