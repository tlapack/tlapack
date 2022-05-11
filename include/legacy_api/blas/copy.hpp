// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_COPY_HH__
#define __TLAPACK_LEGACY_COPY_HH__

#include "legacy_api/base/utils.hpp"
#include "legacy_api/base/types.hpp"
#include "blas/copy.hpp"

namespace tlapack {

/**
 * Copy vector, $y = x$.
 * 
 * Wrapper to copy( const vectorX_t& x, vectorY_t& y ).
 *
 * @param[in] n
 *     Number of elements in x and y. n >= 0.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[out] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup copy
 */
template< typename TX, typename TY >
void copy(
    idx_t n,
    TX const *x, int_t incx,
    TY       *y, int_t incy )
{    
    tblas_error_if( incx == 0 );
    tblas_error_if( incy == 0 );

    // quick return
    if( n <= 0 ) return;
    
    tlapack_expr_with_2vectors(
        _x, TX, n, x, incx,
        _y, TY, n, y, incy,
        return copy( _x, _y )
    );
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_LEGACY_COPY_HH__
