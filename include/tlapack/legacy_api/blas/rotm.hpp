// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_ROTM_HH
#define TLAPACK_LEGACY_ROTM_HH

#include "tlapack/legacy_api/base/utils.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/blas/rotm.hpp"

namespace tlapack {

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
 * @param[in,out] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[in,out] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @param[in] param
 *     Array of length 5 giving parameters of modified plane rotation.
 *
 * @ingroup legacy_blas
 */
template< typename TX, typename TY >
void rotm(
    idx_t n,
    TX *x, int_t incx,
    TY *y, int_t incy,
    scalar_type<TX, TY> const param[5] )
{
    // constants
    const int flag = (int) param[0];
    auto h = &param[1];

    // check arguments
    tlapack_check_false( incx == 0 );
    tlapack_check_false( incy == 0 );
    tlapack_check_false( flag < -2 && flag > 1 );

    // quick return
    if( n <= 0 ) return;
    
    tlapack_expr_with_2vectors(
        x_, TX, n, x, incx,
        y_, TY, n, y, incy,
        if      ( flag == -2 ) return rotm<-2>(x_,y_,h);
        else if ( flag == -1 ) return rotm<-1>(x_,y_,h);
        else if ( flag ==  0 ) return rotm< 0>(x_,y_,h);
        else                   return rotm< 1>(x_,y_,h);
    );
}

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_LEGACY_ROTM_HH
