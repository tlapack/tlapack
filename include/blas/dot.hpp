// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DOT_HH
#define BLAS_DOT_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * @return dot product, $x^H y$.
 * @see dotu for unconjugated version, $x^T y$.
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
 * @ingroup dot
 */
template< class vectorX_t, class vectorY_t >
auto dot( const vectorX_t& x, const vectorY_t& y )
{
    using T = scalar_type<
        type_t< vectorX_t >,
        type_t< vectorY_t >
    >;
    using idx_t = size_type< vectorX_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    blas_error_if( size(y) < n );

    T result( 0.0 );
    for (idx_t i = 0; i < n; ++i)
        result += conj(x[i]) * y[i];

    return result;
}

}  // namespace blas

#endif        //  #ifndef BLAS_DOT_HH
