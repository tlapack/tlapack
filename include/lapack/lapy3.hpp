/// @file lapy3.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lapy3.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LAPY3_HH__
#define __LAPY3_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

namespace lapack {

/** Finds $\sqrt{x^2+y^2+z^2}$, taking care not to cause unnecessary overflow or unnecessary underflow.
 * 
 * @return $\sqrt{x^2+y^2+z^2}$
 *
 * @tparam real_t Floating-point type.
 * @param[in] x scalar value x
 * @param[in] y scalar value y
 * @param[in] z scalar value y
 * 
 * @ingroup auxiliary
 */
template< class TX, class TY, class TZ,
    enable_if_t<(
    /* Requires: */
        ! is_complex<TX>::value &&
        ! is_complex<TY>::value &&
        ! is_complex<TZ>::value
    ), int > = 0 >
real_type<TX,TY,TZ> lapy3(
    const TX& x, const TY& y, const TZ& z )
{
    // using
    using real_t = real_type<TX,TY,TZ>;
    using blas::abs;
    using blas::sqrt;
    using blas::max;

    // constants
    const real_t zero( 0 );
    const auto xabs = abs(x);
    const auto yabs = abs(y);
    const auto zabs = abs(z);
    const auto w = max( xabs, yabs, zabs );

    return ( w == zero )
        // W can be zero for max(0,nan,0)
        // adding all three entries together will make sure
        // NaN will not disappear.
        ? xabs + yabs + zabs
        : w * sqrt(
            (xabs/w)*(xabs/w) + (yabs/w)*(yabs/w) + (zabs/w)*(zabs/w) );
}

}

#endif // __LAPY3_HH__