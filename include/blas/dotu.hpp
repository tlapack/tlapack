// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_DOTU_HH__
#define __TLAPACK_BLAS_DOTU_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * @return unconjugated dot product, $x^T y$.
 * @see dot for conjugated version, $x^H y$.
 *
 * @param[in] x A n-element vector.
 * @param[in] y A n-element vector.
 *
 * @ingroup dotu
 */
template< class vectorX_t, class vectorY_t,
    class T = type_t<vectorY_t>,
    disable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
auto dotu( const vectorX_t& x, const vectorY_t& y )
{
    using return_t = scalar_type<
        type_t< vectorX_t >,
        type_t< vectorY_t >
    >;
    using idx_t = size_type< vectorX_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    tlapack_check_false( size(y) != n );

    return_t result( 0.0 );
    for (idx_t i = 0; i < n; ++i)
        result += x[i] * y[i];

    return result;
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_DOTU_HH__
