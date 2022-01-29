// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_BLAS_SCAL_HH
#define SLATE_BLAS_SCAL_HH

#include "blas/utils.hpp"
#include "blas/scal.hpp"

namespace blas {

/**
 * Scale vector by constant, $x = \alpha x$.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] n
 *     Number of elements in x. n >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*incx + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx > 0.
 *
 * @ingroup scal
 */
template< typename TA, typename TX >
void scal(
    blas::idx_t n,
    const TA& alpha,
    TX* x, blas::int_t incx )
{
    using internal::vector;

    // check arguments
    blas_error_if( incx <= 0 );

    auto _x = vector<TX>( x, n, incx );
    return scal( alpha, _x );
}

}  // namespace blas

#endif        //  #ifndef SLATE_BLAS_SCAL_HH
