// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_GEMM_HH__
#define __TLAPACK_BLAS_GEMM_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * General matrix-matrix multiply:
 * \[
 *     C := \alpha op(A) \times op(B) + \beta C,
 * \]
 * where $op(X)$ is one of
 *     $op(X) = X$,
 *     $op(X) = X^T$, or
 *     $op(X) = X^H$,
 * alpha and beta are scalars, and A, B, and C are matrices, with
 * $op(A)$ an m-by-k matrix, $op(B)$ a k-by-n matrix, and C an m-by-n matrix.
 *
 * @param[in] transA
 *     The operation $op(A)$ to be used:
 *     - Op::NoTrans:   $op(A) = A$.
 *     - Op::Trans:     $op(A) = A^T$.
 *     - Op::ConjTrans: $op(A) = A^H$.
 *
 * @param[in] transB
 *     The operation $op(B)$ to be used:
 *     - Op::NoTrans:   $op(B) = B$.
 *     - Op::Trans:     $op(B) = B^T$.
 *     - Op::ConjTrans: $op(B) = B^H$.
 *
 * @param[in] alpha Scalar.
 * @param[in] A $op(A)$ is an m-by-k matrix.
 * @param[in] B $op(B)$ is an k-by-n matrix.
 * @param[in] beta Scalar.
 * @param[in,out] C A m-by-n matrix.
 * 
 * @ingroup gemm
 */
template<
    class matrixA_t,
    class matrixB_t,
    class matrixC_t,
    class alpha_t,
    class beta_t,
    class T  = alpha_t,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >,
        pair< beta_t,    T >
    > = 0
>
void gemm(
    Op transA,
    Op transB,
    const alpha_t& alpha,
    const matrixA_t& A,
    const matrixB_t& B,
    const beta_t& beta,
    matrixC_t& C )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TB    = type_t< matrixB_t >;
    using idx_t = size_type< matrixA_t >;

    // using
    using scalar_t = scalar_type<TA,TB>;

    // constants
    const idx_t m = (transA == Op::NoTrans) ? nrows(A) : ncols(A);
    const idx_t n = (transB == Op::NoTrans) ? ncols(B) : nrows(B);
    const idx_t k = (transA == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    tlapack_check_false( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    tlapack_check_false( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    tlapack_check_false( nrows(C) != m );
    tlapack_check_false( ncols(C) != n );
    tlapack_check_false( ((transB == Op::NoTrans) ? nrows(B) : ncols(B)) != k );

    tlapack_check_false( access_denied( dense, read_policy(A) ) );
    tlapack_check_false( access_denied( dense, read_policy(B) ) );
    tlapack_check_false( access_denied( dense, write_policy(C) ) );

    if (transA == Op::NoTrans) {
        if (transB == Op::NoTrans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    C(i,j) *= beta;
                for(idx_t l = 0; l < k; ++l) {
                    const auto alphaTimesblj = alpha*B(l,j);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += A(i,l)*alphaTimesblj;
                }
            }
        }
        else if (transB == Op::Trans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    C(i,j) *= beta;
                for(idx_t l = 0; l < k; ++l) {
                    const auto alphaTimesbjl = alpha*B(j,l);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += A(i,l)*alphaTimesbjl;
                }
            }
        }
        else { // transB == Op::ConjTrans
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    C(i,j) *= beta;
                for(idx_t l = 0; l < k; ++l) {
                    const auto alphaTimesbjl = alpha*conj(B(j,l));
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += A(i,l)*alphaTimesbjl;
                }
            }
        }
    }
    else if (transA == Op::Trans) {
        if (transB == Op::NoTrans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*B(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*B(j,l);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else { // transB == Op::ConjTrans
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*conj(B(j,l));
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
    }
    else { // transA == Op::ConjTrans
        if (transB == Op::NoTrans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += conj(A(l,i))*B(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += conj(A(l,i))*B(j,l);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else { // transB == Op::ConjTrans
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*B(j,l); // little improvement here
                    C(i,j) = alpha*conj(sum) + beta*C(i,j);
                }
            }
        }
    }
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_GEMM_HH__
