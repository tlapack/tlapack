/// @file gemm.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_GEMM_HH
#define TLAPACK_BLAS_GEMM_HH
#define MIXED_PREC

#include "tlapack/base/utils.hpp"


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
 * @ingroup blas3
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixC_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<matrixC_t, T>,
                                     pair<alpha_t, T>,
                                     pair<beta_t, T> > = 0>
void gemm(Op transA,
          Op transB,
          const alpha_t& alpha,
          const matrixA_t& A,
          const matrixB_t& B,
          const beta_t& beta,
          matrixC_t& C)
{
    // data traits
    using TA = type_t<matrixA_t>;
    using TB = type_t<matrixB_t>;
    using idx_t = size_type<matrixA_t>;

    // constants
    const idx_t m = (transA == Op::NoTrans) ? nrows(A) : ncols(A);
    const idx_t n = (transB == Op::NoTrans) ? ncols(B) : nrows(B);
    const idx_t k = (transA == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    tlapack_check_false(transA != Op::NoTrans && transA != Op::Trans &&
                        transA != Op::ConjTrans);
    tlapack_check_false(transB != Op::NoTrans && transB != Op::Trans &&
                        transB != Op::ConjTrans);
    tlapack_check_false((idx_t)nrows(C) != m);
    tlapack_check_false((idx_t)ncols(C) != n);
    tlapack_check_false(
        (idx_t)((transB == Op::NoTrans) ? nrows(B) : ncols(B)) != k);

    #ifdef MIXED_PREC
    std::vector<float> MixedMat_(m * n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            MixedMat_[m*j + i] = float(C(i,j));
        }
    }
    #endif
    bool on = true;
    if (transA == Op::NoTrans) {
        using scalar_t = scalar_type<alpha_t, TB>;

        if (transB == Op::NoTrans) {
            #ifdef MIXED_PREC
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i)
                    MixedMat_[m*j + i] *= float(beta);
                for (idx_t l = 0; l < k; ++l) {
                    const scalar_t alphaTimesblj = alpha * B(l, j);
                    for (idx_t i = 0; i < m; ++i)
                        MixedMat_[m*j + i] = sadd(MixedMat_[m*j + i],float(A(i, l) * alphaTimesblj),on);
                }
            }
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    C(i,j) = TA(MixedMat_[m*j + i]);
                }
            }
            #else
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (idx_t l = 0; l < k; ++l) {
                    const scalar_t alphaTimesblj = alpha * B(l, j);
                    for (idx_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l) * alphaTimesblj;
                }
            }

            #endif

        }
        else if (transB == Op::Trans) {
           
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (idx_t l = 0; l < k; ++l) {
                    const scalar_t alphaTimesbjl = alpha * B(j, l);
                    for (idx_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l) * alphaTimesbjl;
                }
            }
        }
        else {  // transB == Op::ConjTrans
    
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (idx_t l = 0; l < k; ++l) {
                    const scalar_t alphaTimesbjl = alpha * conj(B(j, l));
                    for (idx_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l) * alphaTimesbjl;
                }
            }
        }
    }
    else if (transA == Op::Trans) {
        using scalar_t = scalar_type<TA, TB>;

        if (transB == Op::NoTrans) {
          
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * B(l, j);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else if (transB == Op::Trans) {
      
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * B(j, l);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else {  // transB == Op::ConjTrans
      
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * conj(B(j, l));
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
    }
    else {  // transA == Op::ConjTrans

        using scalar_t = scalar_type<TA, TB>;

        if (transB == Op::NoTrans) {
        
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += conj(A(l, i)) * B(l, j);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else if (transB == Op::Trans) {
            
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += conj(A(l, i)) * B(j, l);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else {  // transB == Op::ConjTrans
     
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * B(j, l);  // little improvement here
                    C(i, j) = alpha * conj(sum) + beta * C(i, j);
                }
            }
        }
    }
}

#ifdef TLAPACK_USE_LAPACKPP

/**
 * General matrix-matrix multiply.
 *
 * Wrapper to optimized BLAS.
 *
 * @see gemm(
    Op transA,
    Op transB,
    const alpha_t& alpha,
    const matrixA_t& A,
    const matrixB_t& B,
    const beta_t& beta,
    matrixC_t& C )
*
* @ingroup blas3
*/
template <TLAPACK_LEGACY_MATRIX matrixA_t,
          TLAPACK_LEGACY_MATRIX matrixB_t,
          TLAPACK_LEGACY_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixC_t>,
          enable_if_allow_optblas_t<pair<matrixA_t, T>,
                                    pair<matrixB_t, T>,
                                    pair<matrixC_t, T>,
                                    pair<alpha_t, T>,
                                    pair<beta_t, T> > = 0>
void gemm(Op transA,
          Op transB,
          const alpha_t alpha,
          const matrixA_t& A,
          const matrixB_t& B,
          const beta_t beta,
          matrixC_t& C)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto B_ = legacy_matrix(B);
    auto C_ = legacy_matrix(C);

    // Constants to forward
    constexpr Layout L = layout<matrixC_t>;
    const auto& m = C_.m;
    const auto& n = C_.n;
    const auto& k = (transA == Op::NoTrans) ? A_.n : A_.m;

    // Warnings for NaNs and Infs
    if (alpha == alpha_t(0))
        tlapack_warning(
            -3, "Infs and NaNs in A or B will not propagate to C on output");
    if (beta == beta_t(0) && !is_same_v<beta_t, StrongZero>)
        tlapack_warning(
            -6,
            "Infs and NaNs in C on input will not propagate to C on output");

    return ::blas::gemm((::blas::Layout)L, (::blas::Op)transA,
                        (::blas::Op)transB, m, n, k, alpha, A_.ptr, A_.ldim,
                        B_.ptr, B_.ldim, (T)beta, C_.ptr, C_.ldim);
}

#endif

/**
 * General matrix-matrix multiply:
 * \[
 *     C := \alpha op(A) \times op(B),
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
 * @param[out] C A m-by-n matrix.
 *
 * @ingroup blas3
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t>
void gemm(Op transA,
          Op transB,
          const alpha_t& alpha,
          const matrixA_t& A,
          const matrixB_t& B,
          matrixC_t& C)
{
    return gemm(transA, transB, alpha, A, B, StrongZero(), C);
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_GEMM_HH