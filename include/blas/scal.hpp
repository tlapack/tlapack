// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SCAL_HH
#define BLAS_SCAL_HH

#include "blas/utils.hpp"

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
template< class vector_t, class alpha_t >
void scal( const alpha_t& alpha, vector_t& x )
{
    using idx_t = size_type< vector_t >;

    // constants
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        x[i] *= alpha;
}

}  // namespace blas

#endif        //  #ifndef BLAS_SCAL_HH
