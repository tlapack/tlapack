/// @file syr2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_SYR2_HH
#define TLAPACK_BLAS_SYR2_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Symmetric matrix rank-2 update:
 * \[
 *     A := \alpha x y^T + \alpha y x^T + A,
 * \]
 * where alpha is a scalar, x and y are vectors,
 * and A is an n-by-n symmetric matrix.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed from symmetry.
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Scalar.
 * @param[in] x A n-element vector.
 * @param[in] y A n-element vector.
 * @param[in,out] A A n-by-n symmetric matrix.
 *
 * @ingroup blas2
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorY_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<matrixA_t>,
          disable_if_allow_optblas_t<pair<alpha_t, T>,
                                     pair<matrixA_t, T>,
                                     pair<vectorX_t, T>,
                                     pair<vectorY_t, T> > = 0>
void syr2(Uplo uplo,
          const alpha_t& alpha,
          const vectorX_t& x,
          const vectorY_t& y,
          matrixA_t& A)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using TX = type_t<vectorX_t>;
    using TY = type_t<vectorY_t>;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(size(x) != n);
    tlapack_check_false(size(y) != n);
    tlapack_check_false(ncols(A) != n);

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            const scalar_type<alpha_t, TY> tmp1 = alpha * y[j];
            const scalar_type<alpha_t, TX> tmp2 = alpha * x[j];
            for (idx_t i = 0; i <= j; ++i)
                A(i, j) += x[i] * tmp1 + y[i] * tmp2;
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            const scalar_type<alpha_t, TY> tmp1 = alpha * y[j];
            const scalar_type<alpha_t, TX> tmp2 = alpha * x[j];
            for (idx_t i = j; i < n; ++i)
                A(i, j) += x[i] * tmp1 + y[i] * tmp2;
        }
    }
}

#ifdef USE_LAPACKPP_WRAPPERS

template <TLAPACK_LEGACY_MATRIX matrixA_t,
          TLAPACK_LEGACY_VECTOR vectorX_t,
          TLAPACK_LEGACY_VECTOR vectorY_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<matrixA_t>,
          enable_if_allow_optblas_t<pair<alpha_t, T>,
                                    pair<matrixA_t, T>,
                                    pair<vectorX_t, T>,
                                    pair<vectorY_t, T> > = 0>
inline void syr2(Uplo uplo,
                 const alpha_t alpha,
                 const vectorX_t& x,
                 const vectorY_t& y,
                 matrixA_t& A)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    constexpr Layout L = layout<matrixA_t>;
    const auto& n = A_.n;

    return ::blas::syr2((::blas::Layout)L, (::blas::Uplo)uplo, n, alpha, x_.ptr,
                        x_.inc, y_.ptr, y_.inc, A_.ptr, A_.ldim);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_SYR2_HH
