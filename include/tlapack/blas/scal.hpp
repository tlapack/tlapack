/// @file scal.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_SCAL_HH
#define TLAPACK_BLAS_SCAL_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Scale vector by constant, $x := \alpha x$.
 *
 * @param[in] alpha Scalar.
 * @param[in,out] x A n-element vector.
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vector_t,
          class alpha_t,
          class T = type_t<vector_t>,
          disable_if_allow_optblas_t<pair<alpha_t, T>, pair<vector_t, T> > = 0>
void scal(const alpha_t& alpha, vector_t& x)
{
    using idx_t = size_type<vector_t>;

    // constants
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        x[i] *= alpha;
}

#ifdef USE_LAPACKPP_WRAPPERS

template <TLAPACK_VECTOR vector_t,
          class alpha_t,
          class T = type_t<vector_t>,
          enable_if_allow_optblas_t<pair<alpha_t, T>, pair<vector_t, T> > = 0>
inline void scal(const alpha_t alpha, vector_t& x)
{
    // Legacy objects
    auto x_ = legacy_vector(x);

    // Constants to forward
    const auto& n = x_.n;

    return ::blas::scal(n, alpha, x_.ptr, x_.inc);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_SCAL_HH
