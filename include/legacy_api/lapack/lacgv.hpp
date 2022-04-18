/// @file lacgv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lacgv.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LACGV_HH__
#define __LACGV_HH__

#include "lapack/types.hpp"

namespace lapack {

/** Conjugates a real vector of length n.
 *
 * Dummy inline function that do nothing.
 *
 * @param[in] n
 *     The length of the vector x. n >= 0.
 *
 * @param[in,out] x
 *     The vector x of length n, stored in an array of length 1+(n-1)*abs(incx).
 *     On entry, the vector of length n to be conjugated.
 *     On exit, x is overwritten with conj(x).
 *
 * @param[in] incx
 *     The spacing between successive elements of x.
 *
 * @ingroup auxiliary
 */
template< typename real_t >
inline void lacgv( idx_t n, real_t* x, int_t incx )
{}

/** Conjugates a complex vector of length n.
 *
 * Real precisions are dummy inline functions that do nothing.
 *
 * @param[in] n
 *     The length of the vector x. n >= 0.
 *
 * @param[in,out] x
 *     The vector x of length n, stored in an array of length 1+(n-1)*abs(incx).
 *     On entry, the vector of length n to be conjugated.
 *     On exit, x is overwritten with conj(x).
 *
 * @param[in] incx
 *     The spacing between successive elements of x.
 *
 * @ingroup auxiliary
 */
template< typename T >
void lacgv(
    idx_t n, std::complex<T>* x, int_t incx )
{
    using blas::conj;
    
    if( incx == 1 ) {
        for(idx_t i = 0; i < n; ++i)
            x[i] = conj( x[i] );
    }
    else {
        idx_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (idx_t i = 0; i < n; ++i) {
            x[ix] = conj( x[ix] );
            ix += incx;
        }
    }
}

}

#endif // __LACGV_HH__
