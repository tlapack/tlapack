/// @file ladiv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ladiv.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LADIV_HH__
#define __LADIV_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

namespace lapack {

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
    using blas::abs;

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
template< typename real_t >
inline std::complex<real_t> ladiv(
    const std::complex<real_t>& x,
    const std::complex<real_t>& y )
{
    using blas::real;
    using blas::imag;

    real_t zr, zi;
    ladiv( real(x), imag(x), real(y), imag(y), zr, zi );
    
    return std::complex<real_t>( zr, zi );
}

}

#endif // __LADIV_HH__
