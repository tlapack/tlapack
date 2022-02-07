// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_BLAS_DOTU_HH
#define SLATE_BLAS_DOTU_HH

#include "blas/utils.hpp"
#include "blas/dotu.hpp"

namespace blas {

/**
 * @return unconjugated dot product, $x^T y$.
 * @see dot for conjugated version, $x^H y$.
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
 * @param[in] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup dotu
 */
template< typename TX, typename TY >
scalar_type<TX, TY> dotu(
    blas::idx_t n,
    TX const *x, blas::int_t incx,
    TY const *y, blas::int_t incy )
{
    using internal::vector;

    // check arguments
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // Views
    const auto _x = vector<TX>(
        (TX*) &x[(incx > 0 ? 0 : (-n + 1)*incx)],
        n, incx );
    const auto _y = vector<TY>(
        (TY*) &y[(incy > 0 ? 0 : (-n + 1)*incy)],
        n, incy );

    return dotu( _x, _y );
}

}  // namespace blas

#endif        //  #ifndef SLATE_BLAS_DOTU_HH
