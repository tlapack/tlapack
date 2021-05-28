// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Adapted from https://github.com/langou/latl/blob/master/include/lapy3.h
/// @author Rodney James, University of Colorado Denver, USA

#ifndef __LAPY3_HH__
#define __LAPY3_HH__

#include "lapack/types.hpp"

#include "blas/utils.hpp"

namespace lapack {

/** Finds $\sqrt{x^2+y^2+z^2}$, taking care not to cause unnecessary overflow.
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
template< typename real_t >
void lapy3(
    real_t x, real_t y , real_t z )
{
    using blas::abs;
    using blas::sqrt;
    using blas::max;

    // constants
    const real_t zero(0.0);
    const real_t xabs = abs(x);
    const real_t yabs = abs(y);
    const real_t zabs = abs(z);
    const real_t w = max( xabs, yabs, zabs );

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