/// @file ladiv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see https://github.com/langou/latl/blob/master/include/ladiv.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LADIV_HH
#define TLAPACK_LADIV_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Performs complex division in real arithmetic.
 * 
 * \[
 *      p + iq = (a + ib) / (c + id)
 * \]
 *
 * @tparam real_t Floating-point type.
 * @param[in] a Real part of numerator.
 * @param[in] b Imaginary part of numerator.
 * @param[in] c Real part of denominator.
 * @param[in] d Imaginary part of denominator.
 * @param[out] p Real part of quotient.
 * @param[out] q Imaginary part of quotient.
 * 
 * @ingroup auxiliary
 */
template< typename real_t,
    enable_if_t<(
    /* Requires: */
        ! is_complex<real_t>::value
    ), int > = 0
>
void ladiv(
    const real_t& a, const real_t& b,
    const real_t& c, const real_t& d,
    real_t &p, real_t &q )
{

    real_t e, f;
    if (abs(d) < abs(c)) {
        e = d / c;
        f = c + d * e;
        p = (a + b * e) / f;
        q = (b - a * e) / f;
    }
    else {
        e = c / d;
        f = c + d * e;
        p = (b + a * e) / f;
        q = (-a + b * e) / f;
    }
}

/** Performs complex division in real arithmetic with complex arguments.
 * 
 * @return x/y
 *
 * @tparam real_t Floating-point type.
 * @param[in] x Complex numerator.
 * @param[in] y Complex denominator.
 * 
 * @ingroup auxiliary
 */
template< typename T, enable_if_t< is_complex<T>::value ,int> = 0 >
inline T ladiv(
    const T& x,
    const T& y )
{
    real_type<T> zr, zi;
    ladiv( real(x), imag(x), real(y), imag(y), zr, zi );
    
    return T( zr, zi );
}

}

#endif // TLAPACK_LADIV_HH
