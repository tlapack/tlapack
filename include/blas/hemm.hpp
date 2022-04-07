// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HEMM_HH
#define BLAS_HEMM_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Hermitian matrix-matrix multiply:
 * \[
 *     C := \alpha A B + \beta C,
 * \]
 * or
 * \[
 *     C := \alpha B A + \beta C,
 * \]
 * where alpha and beta are scalars, A is an m-by-m or n-by-n Hermitian matrix,
 * and B and C are m-by-n matrices.
 *
 * @param[in] side
 *     The side the matrix A appears on:
 *     - Side::Left:  $C = \alpha A B + \beta C$,
 *     - Side::Right: $C = \alpha B A + \beta C$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced:
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Scalar.
 * @param[in] A
 *     - If side = Left:  A m-by-m Hermitian matrix.
 *     - If side = Right: A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 * @param[in] B A m-by-n matrix.
 * @param[in] beta Scalar.
 * @param[in,out] C A m-by-n matrix.
 *
 * @ingroup hemm
 */
template<
    class matrixA_t, class matrixB_t, class matrixC_t, 
    class alpha_t, class beta_t >
void hemm(
    blas::Side side,
    blas::Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t& beta, matrixC_t& C )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TB    = type_t< matrixB_t >;
    using idx_t = size_type< matrixB_t >;

    // using
    using scalar_t = scalar_type<TA,TB>;
            
    // constants
    const idx_t m = nrows(B);
    const idx_t n = ncols(B);

    // check arguments
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    blas_error_if( nrows(A) != ncols(A) );
    blas_error_if( nrows(A) != ((side == Side::Left) ? m : n) );
    blas_error_if( nrows(C) != m );
    blas_error_if( ncols(C) != n );

    blas_error_if( access_denied( uplo,  read_policy(A) ) );
    blas_error_if( access_denied( dense, read_policy(B) ) );
    blas_error_if( access_denied( dense, write_policy(C) ) );

    if (side == Side::Left) {
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {

                    auto alphaTimesBij = alpha*B(i,j);
                    scalar_t sum( 0.0 );

                    for(idx_t k = 0; k < i; ++k) {
                        C(k,j) += A(k,i) * alphaTimesBij;
                        sum += conj( A(k,i) ) * B(k,j);
                    }
                    C(i,j) =
                        beta * C(i,j)
                        + real( A(i,i) ) * alphaTimesBij
                        + alpha * sum;
                }
            }
        }
        else {
            // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = m-1; i != idx_t(-1); --i) {

                    auto alphaTimesBij = alpha*B(i,j);
                    scalar_t sum( 0.0 );

                    for(idx_t k = i+1; k < m; ++k) {
                        C(k,j) += A(k,i) * alphaTimesBij;
                        sum += conj( A(k,i) ) * B(k,j);
                    }
                    C(i,j) =
                        beta * C(i,j)
                        + real( A(i,i) ) * alphaTimesBij
                        + alpha * sum;
                }
            }
        }
    }
    else { // side == Side::Right
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {

                auto alphaTimesAkj = alpha * real( A(j,j) );

                for(idx_t i = 0; i < m; ++i)
                    C(i,j) = beta * C(i,j) + B(i,j) * alphaTimesAkj;

                for(idx_t k = 0; k < j; ++k) {
                    alphaTimesAkj = alpha*A(k,j);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }

                for(idx_t k = j+1; k < n; ++k) {
                    alphaTimesAkj = alpha * conj( A(j,k) );
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }
            }
        }
        else {
            // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {

                auto alphaTimesAkj = alpha * real( A(j,j) );

                for(idx_t i = 0; i < m; ++i)
                    C(i,j) = beta * C(i,j) + B(i,j) * alphaTimesAkj;

                for(idx_t k = 0; k < j; ++k) {
                    alphaTimesAkj = alpha * conj( A(j,k) );
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }

                for(idx_t k = j+1; k < n; ++k) {
                    alphaTimesAkj = alpha*A(k,j);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }
            }
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_HEMM_HH
