// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_SWAP_HH
#define TBLAS_LEGACY_SWAP_HH

#include "blas/utils.hpp"
#include "blas/swap.hpp"

namespace blas {

/**
 * Swap vectors, $x <=> y$.
 *
 * Generic implementation for arbitrary data types.
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
 * @param[in,out] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup swap
 */
template< typename TX, typename TY >
void swap(
    blas::idx_t n,
    TX *x, blas::int_t incx,
    TY *y, blas::int_t incy )
{
    // check arguments
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    tlapack_expr_with_2vectors(
        _x, TX, n, x, incx,
        _y, TY, n, y, incy,
        return blas::swap( _x, _y );
    );
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_SWAP_HH
