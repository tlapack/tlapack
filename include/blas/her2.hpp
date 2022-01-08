// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HER2_HH
#define BLAS_HER2_HH

#include "blas/utils.hpp"
#include "blas/syr2.hpp"

namespace blas {

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
 * @param[in, out] A
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 *
 * @param[in] lda
 *     Leading dimension of A. lda >= max(1, n).
 *
 * @ingroup her2
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t >
void her2(
    blas::Uplo  uplo,
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    const matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( size(y)  != n );
    blas_error_if( nrows(A) != ncols(A) ||
                   nrows(A) != n );

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha * conj( y(j) );
            auto tmp2 = conj( alpha * x(j) );
            for (idx_t i = 0; i < j; ++i)
                A(i,j) += x(i)*tmp1 + y(i)*tmp2;
            A(j,j) = real( A(j,j) ) + real( x(j)*tmp1 + y(j)*tmp2 );
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha * conj( y(j) );
            auto tmp2 = conj( alpha * x(j) );
            A(j,j) = real( A(j,j) ) + real( x(j)*tmp1 + y(j)*tmp2 );
            for (idx_t i = j+1; i < n; ++i)
                A(i,j) += x(i)*tmp1 + y(i)*tmp2;
        }
    }
}

template< typename TA, typename TX, typename TY >
void her2(
    blas::Layout layout,
    blas::Uplo  uplo,
    blas::idx_t n,
    blas::scalar_type<TA, TX, TY> alpha,
    TX const *x, blas::int_t incx,
    TY const *y, blas::int_t incy,
    TA *A, blas::idx_t lda )
{
    typedef blas::scalar_type<TA, TX, TY> scalar_t;
    using blas::internal::colmajor_matrix;
    using blas::internal::vector;

    // constants
    const scalar_t zero( 0.0 );

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );
    blas_error_if( lda < n );

    // quick return
    if (n == 0 || alpha == zero)
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }
    
    // Matrix views
    auto _A = colmajor_matrix<TA>( A, n, n, lda );
    const auto _x = vector<TX>(
        (TX*) &x[(incx > 0 ? 0 : (-n + 1)*incx)],
        n, incx );
    const auto _y = vector<TY>(
        (TY*) &y[(incy > 0 ? 0 : (-n + 1)*incy)],
        n, incy );

    her2( uplo, alpha, _x, _y, _A );
}

}  // namespace blas

#endif        //  #ifndef BLAS_HER2_HH
