// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_ROTMG_HH
#define TLAPACK_LEGACY_ROTMG_HH

#include "legacy_api/base/utils.hpp"
#include "legacy_api/base/types.hpp"
#include "blas/rotmg.hpp"

namespace tlapack {

/**
 * Construct modified (fast) plane rotation, H, that eliminates b, such that
 * \[
 *       \begin{bmatrix} z \\ 0 \end{bmatrix}
 *     = H
 *       \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
 *       \begin{bmatrix} a \\ b \end{bmatrix}.
 * \]
 *
 * @see rotm to apply the rotation.
 *
 * With modified plane rotations, vectors u and v are held in factored form as
 * \[
 *     \begin{bmatrix} u^T \\ v^T \end{bmatrix} =
 *     \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
 *     \begin{bmatrix} x^T \\ y^T \end{bmatrix}.
 * \]
 *
 * Application of H to vectors x and y requires 4n flops (2n mul, 2n add)
 * instead of 6n flops (4n mul, 2n add) as in standard plane rotations.
 *
 * Let param = [ flag, $h_{11}, h_{21}, h_{12}, h_{22}$ ].
 * Then H has one of the following forms:
 *
 * - For flag = -1,
 *     \[
 *         H = \begin{bmatrix}
 *             h_{11}  &  h_{12}
 *         \\  h_{21}  &  h_{22}
 *         \end{bmatrix}
 *     \]
 *
 * - For flag = 0,
 *     \[
 *         H = \begin{bmatrix}
 *             1       &  h_{12}
 *         \\  h_{21}  &  1
 *         \end{bmatrix}
 *     \]
 *
 * - For flag = 1,
 *     \[
 *         H = \begin{bmatrix}
 *             h_{11}  &  1
 *         \\  -1      &  h_{22}
 *         \end{bmatrix}
 *     \]
 *
 * - For flag = -2,
 *     \[
 *         H = \begin{bmatrix}
 *             1  &  0
 *         \\  0  &  1
 *         \end{bmatrix}
 *     \]
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in,out] d1
 *     sqrt(d1) is scaling factor for vector x.
 *
 * @param[in,out] d2
 *     sqrt(d2) is scaling factor for vector y.
 *
 * @param[in,out] a
 *     On entry, scalar a. On exit, set to z.
 *
 * @param[in] b
 *     On entry, scalar b.
 *
 * @param[out] param
 *     Array of length 5 giving parameters of modified plane rotation,
 *     as described above.
 *
 * @details
 *
 * Hammarling, Sven. A note on modifications to the Givens plane rotation.
 * IMA Journal of Applied Mathematics, 13:215-218, 1974.
 * http://dx.doi.org/10.1093/imamat/13.2.215
 * (Note the notation swaps u <=> x, v <=> y, d_i -> l_i.)
 *
 * @ingroup rotmg
 */
template< typename real_t >
void rotmg(
    real_t *d1,
    real_t *d2,
    real_t *a,
    real_t  b,
    real_t param[5] )
{
    // check arguments
    tlapack_check_false( *d1 <= 0 );

    param[0] = rotmg(*d1,*d2,*a,b,&param[1]);
}

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_LEGACY_ROTMG_HH
