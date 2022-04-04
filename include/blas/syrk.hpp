// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SYRK_HH
#define BLAS_SYRK_HH

#include "blas/utils.hpp"

namespace blas {

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
    disable_if_allow_optblas_t<matrixA_t, matrixC_t, alpha_t, beta_t> = 0
>
void syrk(
    blas::Uplo uplo,
    blas::Op trans,
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
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans );
    blas_error_if( nrows(C) != ncols(C) );
    blas_error_if( nrows(C) != n );

    blas_error_if( access_denied( dense, read_policy(A) ) );
    blas_error_if( access_denied( uplo, write_policy(C) ) );

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

template<
    class matrixA_t, class matrixC_t, 
    class alpha_t, class beta_t,
    enable_if_allow_optblas_t<matrixA_t, matrixC_t, alpha_t, beta_t> = 0
>
void syrk(
    blas::Uplo uplo,
    blas::Op trans,
    const alpha_t alpha, const matrixA_t& A,
    const beta_t beta, matrixC_t& C )
{
    // Legacy objects
    auto _A = legacy_matrix(A);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& n = _C.n;
    const auto& k = (trans == Op::NoTrans) ? _A.n : _A.m;

    syrk(
        _A.layout, uplo, trans, 
        n, k,
        alpha,
        _A.ptr, _A.ldim,
        beta,
        _C.ptr, _C.ldim );
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYRK_HH
