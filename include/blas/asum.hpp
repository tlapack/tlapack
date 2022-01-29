// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ASUM_HH
#define BLAS_ASUM_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * @return 1-norm of vector,
 *     $|| Re(x) ||_1 + || Im(x) ||_1
 *         = \sum_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] n
 *     Number of elements in x. n >= 0.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*incx + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx > 0.
 *
 * @ingroup asum
 */
template< class vector_t >
real_type< type_t< vector_t > >
asum( vector_t const& x )
{
    using T      = type_t< vector_t >;
    using idx_t  = size_type< vector_t >;
    using real_t = real_type< T >;

    // constants
    const idx_t n = size(x);

    real_t result = 0;
    for (idx_t i = 0; i < n; ++i)
        result += abs1( x(i) );

    return result;
}

}  // namespace blas

#endif        //  #ifndef BLAS_ASUM_HH
