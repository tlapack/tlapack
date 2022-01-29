// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HER_HH
#define BLAS_HER_HH

#include "blas/utils.hpp"
#include "blas/syr.hpp"

namespace blas {

/**
 * Hermitian matrix rank-1 update:
 * \[
 *     A = \alpha x x^H + A,
 * \]
 * where alpha is a scalar, x is a vector,
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
 * @param[in, out] A
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 *
 * @param[in] lda
 *     Leading dimension of A. lda >= max(1, n).
 *
 * @ingroup her
 */
template< class matrixA_t, class vectorX_t, class alpha_t,
    enable_if_t<(
    /* Requires: */
        ! is_complex<alpha_t>::value
    ), int > = 0
>
void her(
    blas::Uplo uplo,
    const alpha_t& alpha,  // zher takes double alpha; use real
    const vectorX_t& x,
    matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( nrows(A) != ncols(A) ||
                   nrows(A) != n );

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp = alpha * conj( x(j) );
            for (idx_t i = 0; i < j; ++i)
                A(i,j) += x(i) * tmp;
            A(j,j) = real( A(j,j) ) + real( x(j) * tmp );
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp = alpha * conj( x(j) );
            A(j,j) = real( A(j,j) ) + real( tmp * x(j) );
            for (idx_t i = j+1; i < n; ++i)
                A(i,j) += x(i) * tmp;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_HER_HH
