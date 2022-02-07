// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HEMV_HH
#define BLAS_HEMV_HH

#include "blas/utils.hpp"
#include "blas/symv.hpp"

namespace blas {

/**
 * Hermitian matrix-vector multiply:
 * \[
 *     y = \alpha A x + \beta y,
 * \]
 * where alpha and beta are scalars, x and y are vectors,
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
 *     Scalar alpha. If alpha is zero, A and x are not accessed.
 *
 * @param[in] A
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *     Imaginary parts of the diagonal elements need not be set,
 *     and are assumed to be zero.
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
 * @ingroup hemv
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t, 
    class alpha_t, class beta_t >
void hemv(
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
                y[i] = beta_t(0);
        }
        else {
            for (idx_t i = 0; i < n; ++i)
                y[i] *= beta;
        }
    }

    if (uplo == Uplo::Upper) {
        // A is stored in upper triangle
        // form y += alpha * A * x
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha*x[j];
            auto tmp2 = scalar_t(0);
            for (idx_t i = 0; i < j; ++i) {
                y[i] += tmp1 * A(i,j);
                tmp2 += conj( A(i,j) ) * x[i];
            }
            y[j] += tmp1 * real( A(j,j) ) + alpha * tmp2;
        }
    }
    else if (uplo == Uplo::Lower) {
        // A is stored in lower triangle
        // form y += alpha * A * x
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha*x[j];
            auto tmp2 = scalar_t(0);
            for (idx_t i = j+1; i < n; ++i) {
                y[i] += tmp1 * A(i,j);
                tmp2 += conj( A(i,j) ) * x[i];
            }
            y[j] += tmp1 * real( A(j,j) ) + alpha * tmp2;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_HEMV_HH
