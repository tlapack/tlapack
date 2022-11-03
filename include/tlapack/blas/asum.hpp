// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_ASUM_HH
#define TLAPACK_BLAS_ASUM_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * @return 1-norm of vector,
 *     $|| Re(x) ||_1 + || Im(x) ||_1
 *         := \sum_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
 * 
 * @param[in] x     A n-element vector.
 *
 * @ingroup asum
 */
template< class vector_t,
    disable_if_allow_optblas_t< vector_t > = 0
>
auto asum( vector_t const& x )
{
    using T      = type_t< vector_t >;
    using idx_t  = size_type< vector_t >;
    using real_t = real_type< T >;

    // constants
    const idx_t n = size(x);

    real_t result = 0;
    for (idx_t i = 0; i < n; ++i)
        result += abs1( x[i] );

    return result;
}

#ifdef USE_LAPACKPP_WRAPPERS

    template< class vector_t,
        enable_if_allow_optblas_t< vector_t > = 0
    >
    inline
    auto asum( vector_t const& x )
    {
        auto x_ = legacy_vector(x);
        return ::blas::asum( x_.n, x_.ptr, x_.inc );
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_ASUM_HH
