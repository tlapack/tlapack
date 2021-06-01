// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Adapted from https://github.com/langou/latl/blob/master/include/ladiv.h
/// @author Rodney James, University of Colorado Denver, USA

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
template< typename real_t >
void ladiv(
    real_t a, real_t b, real_t c, real_t d,
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
    std::complex<real_t> x, std::complex<real_t> y )
{
    using blas::real;
    using blas::imag;

    real_t zr, zi;
    ladiv( real(x), imag(x), real(y), imag(y), zr, zi );
    
    return std::complex<real_t>( zr, zi );
}

}

#endif // __LADIV_HH__