// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_COPY_HH
#define BLAS_COPY_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Copy vector, $y := x$.
 *
 * @param[in]  x A n-element vector x.
 * @param[out] y A vector of at least n elements.
 *
 * @ingroup copy
 */
template< class vectorX_t, class vectorY_t >
void copy( const vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    blas_error_if( size(y) < n );

    for (idx_t i = 0; i < n; ++i)
        y[i] = x[i];
}

}  // namespace blas

#endif        //  #ifndef BLAS_COPY_HH
