/// @file larfg.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larfg.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_LARFG_HH__
#define __TLAPACK_LEGACY_LARFG_HH__

#include "lapack/larfg.hpp"

namespace lapack {

template< typename T >
void larfg(
    blas::idx_t n, T &alpha, T *x, blas::int_t incx, T &tau )
{    
    tlapack_expr_with_vector( _x, T, n-1, x, incx, return larfg( alpha, _x, tau ) );
}

/** Generates a elementary Householder reflection.
 * 
 * @see larfg( blas::idx_t, T &, T *, blas::int_t, T & )
 * 
 * @ingroup auxiliary
 */
template< typename T >
void inline larfg(
    blas::idx_t n, T *alpha, T *x, blas::int_t incx, T *tau )
{
    larfg(n, *alpha, x, incx, *tau);
}

}

#endif // __TLAPACK_LEGACY_LARFG_HH__
