/// @file dot.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_DOT_HH
#define TLAPACK_BLAS_DOT_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * @return dot product, $x^H y$.
 * @see dotu for unconjugated version, $x^T y$.
 *
 * @param[in] x A n-element vector.
 * @param[in] y A n-element vector.
 *
 * @ingroup blas1
 */
template <
    TLAPACK_VECTOR vectorX_t,
    TLAPACK_VECTOR vectorY_t,
    class T = type_t<vectorY_t>,
    disable_if_allow_optblas_t<pair<vectorX_t, T>, pair<vectorY_t, T> > = 0>
auto dot(const vectorX_t& x, const vectorY_t& y)
{
    using return_t = scalar_type<type_t<vectorX_t>, type_t<vectorY_t> >;
    using idx_t = size_type<vectorX_t>;

    // constants
    const idx_t n = size(x);

    // check arguments
    tlapack_check_false(size(y) != n);

    return_t result(0);
    for (idx_t i = 0; i < n; ++i)
        result += conj(x[i]) * y[i];

    return result;
}

#ifdef TLAPACK_USE_LAPACKPP

template <
    TLAPACK_LEGACY_VECTOR vectorX_t,
    TLAPACK_LEGACY_VECTOR vectorY_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<pair<vectorX_t, T>, pair<vectorY_t, T> > = 0>
auto dot(const vectorX_t& x, const vectorY_t& y)
{
    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const auto& n = x_.n;

    return ::blas::dot(n, x_.ptr, x_.inc, y_.ptr, y_.inc);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_DOT_HH
