// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_SYRK_HH__
#define __TLAPACK_BLAS_SYRK_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * Symmetric rank-k update:
 * \[
 *     C := \alpha A A^T + \beta C,
 * \]
 * or
 * \[
 *     C := \alpha A^T A + \beta C,
 * \]
 * where alpha and beta are scalars, C is an n-by-n symmetric matrix,
 * and A is an n-by-k or k-by-n matrix.
 *
 * @param[in] uplo
 *     What part of the matrix C is referenced,
 *     the opposite triangle being assumed from symmetry:
 *     - Uplo::Lower: only the lower triangular part of C is referenced.
 *     - Uplo::Upper: only the upper triangular part of C is referenced.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans: $C = \alpha A A^T + \beta C$.
 *     - Op::Trans:   $C = \alpha A^T A + \beta C$.
 *
 * @param[in] alpha Scalar.
 * @param[in] A A n-by-k matrix.
 *     - If trans = NoTrans: a n-by-k matrix.
 *     - Otherwise:          a k-by-n matrix.
 * @param[in] beta Scalar.
 * @param[in,out] C A n-by-n symmetric matrix.
 *
 * @ingroup syrk
 */
template<
    class matrixA_t, class matrixC_t, 
    class alpha_t, class beta_t,
    class T  = alpha_t,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >,
        pair< beta_t,    T >
    > = 0
>
void syrk(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A,
    const beta_t& beta, matrixC_t& C )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = (trans == Op::NoTrans) ? nrows(A) : ncols(A);
    const idx_t k = (trans == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    tblas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    tblas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans );
    tblas_error_if( nrows(C) != ncols(C) );
    tblas_error_if( nrows(C) != n );

    tblas_error_if( access_denied( dense, read_policy(A) ) );
    tblas_error_if( access_denied( uplo, write_policy(C) ) );

    if (trans == Op::NoTrans) {
        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {

                for(idx_t i = 0; i <= j; ++i)
                    C(i,j) *= beta;

                for(idx_t l = 0; l < k; ++l) {
                    auto alphaAjl = alpha*A(j,l);
                    for(idx_t i = 0; i <= j; ++i)
                        C(i,j) += A(i,l)*alphaAjl;
                }
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {

                for(idx_t i = j; i < n; ++i)
                    C(i,j) *= beta;

                for(idx_t l = 0; l < k; ++l) {
                    auto alphaAjl = alpha * A(j,l);
                    for(idx_t i = j; i < n; ++i)
                        C(i,j) += A(i,l) * alphaAjl;
                }
            }
        }
    }
    else { // trans == Op::Trans
        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i <= j; ++i) {
                    TA sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i) * A(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = j; i < n; ++i) {
                    TA sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum +=  A(l,i) * A(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
    }

    if (uplo == Uplo::General) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = j+1; i < n; ++i)
                C(i,j) = C(j,i);
        }
    }
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_SYRK_HH__
