// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_DOT_HH__
#define __TLAPACK_BLAS_DOT_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * @return dot product, $x^H y$.
 * @see dotu for unconjugated version, $x^T y$.
 *
 * @param[in] x A n-element vector.
 * @param[in] y A n-element vector.
 *
 * @ingroup dot
 */
template< class vectorX_t, class vectorY_t,
    disable_if_allow_optblas_t<
        pair< vectorX_t, type_t< vectorX_t > >,
        pair< vectorY_t, type_t< vectorX_t > >
    > = 0
>
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
    tblas_error_if( size(y) != n );

    T result( 0.0 );
    for (idx_t i = 0; i < n; ++i)
        result += conj(x[i]) * y[i];

    return result;
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_DOT_HH__
