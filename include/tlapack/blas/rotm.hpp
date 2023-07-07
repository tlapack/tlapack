/// @file rotm.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_ROTM_HH
#define TLAPACK_BLAS_ROTM_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Apply modified (fast) plane rotation, H:
 * \[
 *       \begin{bmatrix} x^T \\ y^T \end{bmatrix}
 *     := H
 *       \begin{bmatrix} x^T \\ y^T \end{bmatrix}.
 * \]
 *
 * @see rotmg to generate the rotation, and for fuller description.
 *
 * @tparam flag Defines the format of H.
 * - For flag = -1,
 *     \[
 *         H = \begin{bmatrix}
 *             h_{11}  &  h_{12}
 *         \\  h_{21}  &  h_{22}
 *         \end{bmatrix}
 *     \]
 * - For flag = 0,
 *     \[
 *         H = \begin{bmatrix}
 *             1       &  h_{12}
 *         \\  h_{21}  &  1
 *         \end{bmatrix}
 *     \]
 * - For flag = 1,
 *     \[
 *         H = \begin{bmatrix}
 *             h_{11}  &  1
 *         \\  -1      &  h_{22}
 *         \end{bmatrix}
 *     \]
 * - For flag = -2,
 *     \[
 *         H = \begin{bmatrix}
 *             1  &  0
 *         \\  0  &  1
 *         \end{bmatrix}
 *     \]
 *
 * @param[in,out] x A n-element vector.
 * @param[in,out] y A n-element vector.
 * @param[in]     h 4-element array with the modified plane rotation.
 * \[
 *      h = { h_{11}, h_{21}, h_{12}, h_{22} }.
 * \]
 *
 * @ingroup blas1
 */
template <
    int flag,
    TLAPACK_VECTOR vectorX_t,
    TLAPACK_VECTOR vectorY_t,
    enable_if_t<((-2 <= flag) && (flag <= 1)), int> = 0,
    class T = type_t<vectorX_t>,
    enable_if_t<is_same_v<T, real_type<T> >, int> = 0,
    disable_if_allow_optblas_t<pair<vectorX_t, T>, pair<vectorY_t, T> > = 0>
void rotm(vectorX_t& x, vectorY_t& y, const T h[4])
{
    using idx_t = size_type<vectorX_t>;
    using scalar_t = scalar_type<T, type_t<vectorX_t>, type_t<vectorY_t> >;

    // constants
    const idx_t n = size(x);

    // check arguments
    tlapack_check_false(size(y) != n);

    if (flag == -1) {
        for (idx_t i = 0; i < n; ++i) {
            const scalar_t stmp = h[0] * x[i] + h[2] * y[i];
            y[i] = h[3] * y[i] + h[1] * x[i];
            x[i] = stmp;
        }
    }
    else if (flag == 0) {
        for (idx_t i = 0; i < n; ++i) {
            const scalar_t stmp = x[i] + h[2] * y[i];
            y[i] = y[i] + h[1] * x[i];
            x[i] = stmp;
        }
    }
    else if (flag == 1) {
        for (idx_t i = 0; i < n; ++i) {
            const scalar_t stmp = h[0] * x[i] + y[i];
            y[i] = h[3] * y[i] - x[i];
            x[i] = stmp;
        }
    }
}

#ifdef TLAPACK_USE_LAPACKPP

template <
    int flag,
    TLAPACK_LEGACY_VECTOR vectorX_t,
    TLAPACK_LEGACY_VECTOR vectorY_t,
    enable_if_t<((-2 <= flag) && (flag <= 1)), int> = 0,
    class T = type_t<vectorX_t>,
    enable_if_t<is_same_v<T, real_type<T> >, int> = 0,
    enable_if_allow_optblas_t<pair<vectorX_t, T>, pair<vectorY_t, T> > = 0>
inline void rotm(vectorX_t& x, vectorY_t& y, const T h[4])
{
    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const auto& n = x_.n;
    const T h_[] = {(T)flag, h[0], h[1], h[2], h[3]};

    return ::blas::rotm(n, x_.ptr, x_.inc, y_.ptr, y_.inc, h_);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_ROTM_HH
