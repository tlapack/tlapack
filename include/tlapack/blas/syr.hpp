/// @file syr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_SYR_HH
#define TLAPACK_BLAS_SYR_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Symmetric matrix rank-1 update:
 * \[
 *     A := \alpha x x^T + A,
 * \]
 * where alpha is a scalar, x is a vector,
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
 * @param[in,out] A A n-by-n symmetric matrix.
 *
 * @ingroup blas2
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          class alpha_t,
          class T = type_t<matrixA_t>,
          disable_if_allow_optblas_t<pair<alpha_t, T>,
                                     pair<matrixA_t, T>,
                                     pair<vectorX_t, T> > = 0>
void syr(Uplo uplo, const alpha_t& alpha, const vectorX_t& x, matrixA_t& A)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = scalar_type<alpha_t, type_t<vectorX_t> >;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(size(x) != n);
    tlapack_check_false(ncols(A) != n);

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            const scalar_t tmp = alpha * x[j];
            for (idx_t i = 0; i <= j; ++i)
                A(i, j) += x[i] * tmp;
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            const scalar_t tmp = alpha * x[j];
            for (idx_t i = j; i < n; ++i)
                A(i, j) += x[i] * tmp;
        }
    }
}

#ifdef USE_LAPACKPP_WRAPPERS

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          class alpha_t,
          class T = type_t<matrixA_t>,
          enable_if_allow_optblas_t<pair<alpha_t, T>,
                                    pair<matrixA_t, T>,
                                    pair<vectorX_t, T> > = 0>
inline void syr(Uplo uplo,
                const alpha_t alpha,
                const vectorX_t& x,
                matrixA_t& A)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);

    // Constants to forward
    constexpr Layout L = layout<matrixA_t>;
    const auto& n = A_.n;

    return ::blas::syr((::blas::Layout)L, (::blas::Uplo)uplo, n, alpha, x_.ptr,
                       x_.inc, A_.ptr, A_.ldim);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_SYR_HH
