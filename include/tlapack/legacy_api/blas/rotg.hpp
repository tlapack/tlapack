/// @file rotg.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_ROTG_HH
#define TLAPACK_LEGACY_ROTG_HH

#include "tlapack/blas/rotg.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {

/**
 * Construct plane rotation that eliminates b, such that:
 * \[
 *       \begin{bmatrix} r     \\ 0      \end{bmatrix}
 *     = \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
 *       \begin{bmatrix} a     \\ b      \end{bmatrix}.
 * \]
 *
 * @see rot to apply the rotation.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in,out] a
 *     On entry, scalar a. On exit, set to r.
 *
 * @param[in,out] b
 *     On entry, scalar b. On exit, set to s, 1/c, or 0.
 *
 * @param[out] c
 *     Cosine of rotation; real.
 *
 * @param[out] s
 *     Sine of rotation; scalar.
 *
 * @details
 *
 * Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
 * ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665
 *
 * @ingroup legacy_blas
 */
template <typename T>
inline void rotg(T* a, T* b, real_type<T>* c, T* s)
{
    return rotg(*a, *b, *c, *s);
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_ROTG_HH
