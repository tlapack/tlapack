// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ROTM_HH
#define BLAS_ROTM_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Apply modified (fast) plane rotation, H:
 * \[
 *       \begin{bmatrix} x^T \\ y^T \end{bmatrix}
 *     = H
 *       \begin{bmatrix} x^T \\ y^T \end{bmatrix}.
 * \]
 *
 * @see rotmg to generate the rotation, and for fuller description.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] n
 *     Number of elements in x and y. n >= 0.
 *
 * @param[in, out] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[in, out] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @param[in] param
 *     Array of length 5 giving parameters of modified plane rotation.
 *
 * @ingroup rotm
 */
template<
    int flag,
    class vectorX_t, class vectorY_t, class real_t,
    enable_if_t<((-2 <= flag) && (flag <= 1)), int > = 0
>
void rotm(
    vectorX_t& x, vectorY_t& y,
    const real_t H[4] )
{
    using idx_t = size_type< vectorY_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    blas_error_if( size(y) != n );

    if ( flag == -1 ) {
        for (idx_t i = 0; i < n; ++i) {
            auto stmp = H[0]*x[i] + H[2]*y[i];
            y[i] = H[3]*y[i] + H[1]*x[i];
            x[i] = stmp;
        }
    }
    else if ( flag == 0 ) {
        for (idx_t i = 0; i < n; ++i) {
            auto stmp = x[i] + H[2]*y[i];
            y[i] = y[i] + H[1]*x[i];
            x[i] = stmp;
        }
    }
    else if ( flag == 1 ) {
        for (idx_t i = 0; i < n; ++i) {
            auto stmp = H[0]*x[i] + y[i];
            y[i] = H[3]*y[i] - x[i];
            x[i] = stmp;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTM_HH
