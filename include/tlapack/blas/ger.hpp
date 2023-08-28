/// @file ger.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_GER_HH
#define TLAPACK_BLAS_GER_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * General matrix rank-1 update:
 * \[
 *     A := \alpha x y^H + A,
 * \]
 * where alpha is a scalar, x and y are vectors,
 * and A is an m-by-n matrix.
 *
 * @param[in] alpha Scalar.
 * @param[in] x A m-element vector.
 * @param[in] y A n-element vector.
 * @param[in] A A m-by-n matrix.
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
void ger(const alpha_t& alpha,
         const vectorX_t& x,
         const vectorY_t& y,
         matrixA_t& A)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = scalar_type<alpha_t, type_t<vectorY_t> >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(size(x) != m);
    tlapack_check_false(size(y) != n);

    for (idx_t j = 0; j < n; ++j) {
        const scalar_t tmp = alpha * conj(y[j]);
        for (idx_t i = 0; i < m; ++i)
            A(i, j) += x[i] * tmp;
    }
}

#ifdef TLAPACK_USE_LAPACKPP

template <TLAPACK_LEGACY_MATRIX matrixA_t,
          TLAPACK_LEGACY_VECTOR vectorX_t,
          TLAPACK_LEGACY_VECTOR vectorY_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<matrixA_t>,
          enable_if_allow_optblas_t<pair<alpha_t, T>,
                                    pair<matrixA_t, T>,
                                    pair<vectorX_t, T>,
                                    pair<vectorY_t, T> > = 0>
void ger(const alpha_t alpha,
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
    const auto& m = A_.m;
    const auto& n = A_.n;

    return ::blas::ger((::blas::Layout)L, m, n, alpha, x_.ptr, x_.inc, y_.ptr,
                       y_.inc, A_.ptr, A_.ldim);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_GER_HH
