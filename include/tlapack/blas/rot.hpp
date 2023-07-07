/// @file rot.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_ROT_HH
#define TLAPACK_BLAS_ROT_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Apply plane rotation:
 * \[
 *       \begin{bmatrix} x^T   \\ y^T    \end{bmatrix}
 *     := \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
 *       \begin{bmatrix} x^T   \\ y^T    \end{bmatrix}.
 * \]
 *
 * @see rotg to generate the rotation.
 *
 * @param[in,out] x A n-element vector.
 * @param[in,out] y A n-element vector.
 * @param[in] c     Cosine of rotation; real.
 * @param[in] s     Sine of rotation; scalar.
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorY_t,
          TLAPACK_REAL c_type,
          TLAPACK_SCALAR s_type,
          class T = type_t<vectorX_t>,
          disable_if_allow_optblas_t<pair<vectorX_t, T>,
                                     pair<vectorY_t, T>,
                                     pair<c_type, real_type<T> >,
                                     pair<s_type, real_type<T> > > = 0>
void rot(vectorX_t& x, vectorY_t& y, const c_type& c, const s_type& s)
{
    using idx_t = size_type<vectorX_t>;
    using scalar_t =
        scalar_type<c_type, s_type, type_t<vectorX_t>, type_t<vectorY_t> >;

    // constants
    const idx_t n = size(x);

    // check arguments
    tlapack_check_false(size(y) != n);

    // quick return
    if (n == 0 || (c == c_type(1) && s == s_type(0))) return;

    for (idx_t i = 0; i < n; ++i) {
        const scalar_t stmp = c * x[i] + s * y[i];
        y[i] = c * y[i] - conj(s) * x[i];
        x[i] = stmp;
    }
}

#ifdef TLAPACK_USE_LAPACKPP

template <TLAPACK_LEGACY_VECTOR vectorX_t,
          TLAPACK_LEGACY_VECTOR vectorY_t,
          TLAPACK_REAL c_type,
          TLAPACK_SCALAR s_type,
          class T = type_t<vectorX_t>,
          enable_if_allow_optblas_t<pair<vectorX_t, T>,
                                    pair<vectorY_t, T>,
                                    pair<c_type, real_type<T> >,
                                    pair<s_type, real_type<T> > > = 0>
inline void rot(vectorX_t& x, vectorY_t& y, const c_type c, const s_type s)
{
    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const auto& n = x_.n;

    return ::blas::rot(n, x_.ptr, x_.inc, y_.ptr, y_.inc, c, s);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_ROT_HH
