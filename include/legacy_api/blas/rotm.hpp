// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_ROTM_HH
#define TBLAS_LEGACY_ROTM_HH

#include "blas/utils.hpp"
#include "blas/rotm.hpp"

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
template< typename TX, typename TY >
void rotm(
    blas::idx_t n,
    TX *x, blas::int_t incx,
    TY *y, blas::int_t incy,
    blas::scalar_type<TX, TY> const param[5] )
{
    using internal::vector;

    // constants
    const int flag = (int) param[0];

    // check arguments
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // quick return
    if ( n == 0 || flag == -2 )
        return;

    // Views
    auto _x = vector<TX>(
        &x[(incx > 0 ? 0 : (-n + 1)*incx)],
        n, incx );
    auto _y = vector<TY>(
        &y[(incy > 0 ? 0 : (-n + 1)*incy)],
        n, incy );

    switch (flag) {
    case -2: return rotm<-2>(_x,_y,&param[1]);
    case -1: return rotm<-1>(_x,_y,&param[1]);
    case  0: return rotm< 0>(_x,_y,&param[1]);
    case  1: return rotm< 1>(_x,_y,&param[1]);
    default:
        throw Error("Invalid flag in blas::rotm");
    }
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_ROTM_HH