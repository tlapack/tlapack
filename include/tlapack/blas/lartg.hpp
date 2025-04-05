/// @file lartg.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_LARTG_HH
#define TLAPACK_BLAS_LARTG_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rotg.hpp"

namespace tlapack {

/**
 * Construct plane rotation that eliminates b, such that:
 * \[
 *       \begin{bmatrix} r     \\ 0      \end{bmatrix}
 *     := \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
 *       \begin{bmatrix} a     \\ b      \end{bmatrix}.
 * \]
 *
 * @see rot to apply the rotation.
 *
 * @param[in]   a On entry, scalar a. On exit, set to r.
 * @param[in]   b On entry, scalar b. On exit, set to s, 1/c, or 0.
 * @param[out]  c Cosine of rotation; real.
 * @param[out]  s Sine of rotation.
 * @param[out]  r The nonzero component of the rotated vector.
 *
 * @ingroup blas1
 */
template <typename T, enable_if_t<is_same_v<T, real_type<T> >, int> = 0>
void lartg(const T& a, const T& b, real_type<T>& c, T& s, T& r)
{
    r = a;
    T btemp = b;
    rotg(r, btemp, c, s);
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_LARTG_HH
