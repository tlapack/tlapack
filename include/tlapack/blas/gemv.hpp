/// @file gemv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_GEMV_HH
#define TLAPACK_BLAS_GEMV_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * General matrix-vector multiply:
 * \[
 *     y := \alpha op(A) x + \beta y,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$,
 *     $op(A) = A^H$, or
 *     $op(A) = conj(A)$,
 * alpha and beta are scalars, x and y are vectors, and A is a matrix.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $y = \alpha A   x + \beta y$,
 *     - Op::Trans:     $y = \alpha A^T x + \beta y$,
 *     - Op::ConjTrans: $y = \alpha A^H x + \beta y$,
 *     - Op::Conj:  $y = \alpha conj(A) x + \beta y$.
 *
 * @param[in] alpha Scalar.
 * @param[in] A $op(A)$ is an m-by-n matrix.
 * @param[in] x A n-element vector.
 * @param[in] beta Scalar.
 * @param[in,out] y A m-element vector.
 *
 * @ingroup blas2
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorY_t,
          class alpha_t,
          class beta_t,
          class T = type_t<vectorY_t>,
          disable_if_allow_optblas_t<pair<alpha_t, T>,
                                     pair<matrixA_t, T>,
                                     pair<vectorX_t, T>,
                                     pair<vectorY_t, T>,
                                     pair<beta_t, T> > = 0>
void gemv(Op trans,
          const alpha_t& alpha,
          const matrixA_t& A,
          const vectorX_t& x,
          const beta_t& beta,
          vectorY_t& y)
{
    // data traits
    using TA = type_t<matrixA_t>;
    using TX = type_t<vectorX_t>;
    using idx_t = size_type<matrixA_t>;

    // constants
    const idx_t m =
        (trans == Op::NoTrans || trans == Op::Conj) ? nrows(A) : ncols(A);
    const idx_t n =
        (trans == Op::NoTrans || trans == Op::Conj) ? ncols(A) : nrows(A);

    // check arguments
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans && trans != Op::Conj);
    tlapack_check_false((idx_t)size(x) != n);
    tlapack_check_false((idx_t)size(y) != m);

    // quick return
    if (m == 0 || n == 0) return;

    // form y := beta*y
    for (idx_t i = 0; i < m; ++i)
        y[i] *= beta;

    if (trans == Op::NoTrans) {
        // form y += alpha * A * x
        for (idx_t j = 0; j < n; ++j) {
            const scalar_type<alpha_t, TX> tmp = alpha * x[j];
            for (idx_t i = 0; i < m; ++i) {
                y[i] += tmp * A(i, j);
            }
        }
    }
    else if (trans == Op::Conj) {
        // form y += alpha * conj( A ) * x
        for (idx_t j = 0; j < n; ++j) {
            const scalar_type<alpha_t, TX> tmp = alpha * x[j];
            for (idx_t i = 0; i < m; ++i) {
                y[i] += tmp * conj(A(i, j));
            }
        }
    }
    else if (trans == Op::Trans) {
        // form y += alpha * A^T * x
        for (idx_t i = 0; i < m; ++i) {
            scalar_type<TA, TX> tmp(0);
            for (idx_t j = 0; j < n; ++j) {
                tmp += A(j, i) * x[j];
            }
            y[i] += alpha * tmp;
        }
    }
    else {
        // form y += alpha * A^H * x
        for (idx_t i = 0; i < m; ++i) {
            scalar_type<TA, TX> tmp(0);
            for (idx_t j = 0; j < n; ++j) {
                tmp += conj(A(j, i)) * x[j];
            }
            y[i] += alpha * tmp;
        }
    }
}

#ifdef USE_LAPACKPP_WRAPPERS

/**
 * General matrix-vector multiply.
 *
 * Wrapper to optimized BLAS.
 *
 * @see gemv(
    Op trans,
    const alpha_t& alpha, const matrixA_t& A, const vectorX_t& x,
    const beta_t& beta, vectorY_t& y )
*
* @ingroup blas2
*/
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorY_t,
          class alpha_t,
          class beta_t,
          class T = type_t<vectorY_t>,
          enable_if_allow_optblas_t<pair<alpha_t, T>,
                                    pair<matrixA_t, T>,
                                    pair<vectorX_t, T>,
                                    pair<vectorY_t, T>,
                                    pair<beta_t, T> > = 0>
inline void gemv(Op trans,
                 const alpha_t alpha,
                 const matrixA_t& A,
                 const vectorX_t& x,
                 const beta_t beta,
                 vectorY_t& y)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    constexpr Layout L = layout<matrixA_t>;
    const auto& m = A_.m;
    const auto& n = A_.n;

    // Warnings for NaNs and Infs
    if (alpha == alpha_t(0))
        tlapack_warning(
            -2, "Infs and NaNs in A or x will not propagate to y on output");
    if (beta == beta_t(0) && !is_same_v<beta_t, StrongZero>)
        tlapack_warning(
            -5,
            "Infs and NaNs in y on input will not propagate to y on output");

    return ::blas::gemv((::blas::Layout)L, (::blas::Op)trans, m, n, alpha,
                        A_.ptr, A_.ldim, x_.ptr, x_.inc, (T)beta, y_.ptr,
                        y_.inc);
}

#endif

/**
 * General matrix-vector multiply:
 * \[
 *     y := \alpha op(A) x,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$,
 *     $op(A) = A^H$, or
 *     $op(A) = conj(A)$,
 * alpha and beta are scalars, x and y are vectors, and A is a matrix.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $y = \alpha A   x$,
 *     - Op::Trans:     $y = \alpha A^T x$,
 *     - Op::ConjTrans: $y = \alpha A^H x$,
 *     - Op::Conj:  $y = \alpha conj(A) x$.
 *
 * @param[in] alpha Scalar.
 * @param[in] A $op(A)$ is an m-by-n matrix.
 * @param[in] x A n-element vector.
 * @param[in,out] y A m-element vector.
 *
 * @ingroup blas2
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorY_t,
          class alpha_t>
inline void gemv(Op trans,
                 const alpha_t& alpha,
                 const matrixA_t& A,
                 const vectorX_t& x,
                 vectorY_t& y)
{
    return gemv(trans, alpha, A, x, StrongZero(), y);
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_GEMV_HH
