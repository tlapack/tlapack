// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_ASUM_HH
#define TBLAS_LEGACY_ASUM_HH

#include "blas/utils.hpp"
#include "blas/asum.hpp"

namespace blas {

/**
 * Wrapper to asum( vector_t const& x ).
 * 
 * @return 1-norm of vector,
 *     $|| Re(x) ||_1 + || Im(x) ||_1
 *         = \sum_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
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
template< typename T >
inline
auto asum( idx_t n, T const *x, int_t incx )
{
    blas_error_if( incx <= 0 );
    
    tlapack_expr_with_vector_positiveInc(
        _x, T, n, x, incx,
        return asum( _x )
    );
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_ASUM_HH
