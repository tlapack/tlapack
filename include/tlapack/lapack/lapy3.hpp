/// @file lapy3.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/lapy3.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAPY3_HH
#define TLAPACK_LAPY3_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Finds $\sqrt{x^2+y^2+z^2}$, taking care not to cause unnecessary overflow or
 * unnecessary underflow.
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
template <TLAPACK_REAL TX,
          TLAPACK_REAL TY,
          TLAPACK_REAL TZ,
          enable_if_t<(
                          /* Requires: */
                          is_real<TX>::value && is_real<TY>::value &&
                          is_real<TZ>::value),
                      int> = 0>
real_type<TX, TY, TZ> lapy3(const TX& x, const TY& y, const TZ& z)
{
    // using
    using real_t = real_type<TX, TY, TZ>;

    // constants
    const real_t zero(0);
    const TX xabs = abs(x);
    const TY yabs = abs(y);
    const TZ zabs = abs(z);
    const real_t w = max(xabs, max(yabs, zabs));

    return (w == zero)
               // W can be zero for max(0,nan,0)
               // adding all three entries together will make sure
               // NaN will not disappear.
               ? xabs + yabs + zabs
               : w * sqrt((xabs / w) * (xabs / w) + (yabs / w) * (yabs / w) +
                          (zabs / w) * (zabs / w));
}

}  // namespace tlapack

#endif  // TLAPACK_LAPY3_HH