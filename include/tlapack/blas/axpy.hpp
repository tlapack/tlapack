/// @file axpy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_AXPY_HH
#define TLAPACK_BLAS_AXPY_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Add scaled vector, $y := \alpha x + y$.
 *
 * @param[in] alpha Scalar.
 * @param[in] x     A n-element vector.
 * @param[in,out] y A vector with at least n elements.
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorY_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<vectorY_t>,
          disable_if_allow_optblas_t<pair<alpha_t, T>,
                                     pair<vectorX_t, T>,
                                     pair<vectorY_t, T> > = 0>
void axpy(const alpha_t& alpha, const vectorX_t& x, vectorY_t& y)
{
    using idx_t = size_type<vectorX_t>;

    // constants
    const idx_t n = size(x);

    // check arguments
    tlapack_check_false((idx_t)size(y) < n);

    for (idx_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

#ifdef TLAPACK_USE_LAPACKPP

template <TLAPACK_LEGACY_VECTOR vectorX_t,
          TLAPACK_LEGACY_VECTOR vectorY_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<vectorY_t>,
          enable_if_allow_optblas_t<pair<alpha_t, T>,
                                    pair<vectorX_t, T>,
                                    pair<vectorY_t, T> > = 0>
void axpy(const alpha_t alpha, const vectorX_t& x, vectorY_t& y)
{
    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const auto& n = x_.n;

    // Warnings for NaNs and Infs
    if (alpha == alpha_t(0))
        tlapack_warning(-1,
                        "Infs and NaNs in x will not propagate to y on output");

    return ::blas::axpy(n, alpha, x_.ptr, x_.inc, y_.ptr, y_.inc);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_AXPY_HH
