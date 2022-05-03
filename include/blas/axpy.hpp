// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_AXPY_HH__
#define __TLAPACK_BLAS_AXPY_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * Add scaled vector, $y := \alpha x + y$.
 *
 * @param[in] alpha Scalar.
 * @param[in] x     A n-element vector.
 * @param[in,out] y A vector with at least n elements.
 *
 * @ingroup axpy
 */
template< class vectorX_t, class vectorY_t, class alpha_t,
    disable_if_allow_optblas_t<
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >
    > = 0
>
void axpy(
    const alpha_t& alpha,
    const vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    tlapack_error_if( size(y) < n );

    for (idx_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_AXPY_HH__
