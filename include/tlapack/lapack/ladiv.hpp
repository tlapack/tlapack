/// @file ladiv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LADIV_HH
#define TLAPACK_LADIV_HH

#include "tlapack/base/constants.hpp"
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
template <TLAPACK_REAL real_t,
          enable_if_t<(
                          /* Requires: */
                          is_real<real_t>),
                      int> = 0>
void ladiv(const real_t& a,
           const real_t& b,
           const real_t& c,
           const real_t& d,
           real_t& p,
           real_t& q)
{
    // internal function ladiv2
    auto ladiv2 = [](const real_t& a, const real_t& b, const real_t& c,
                     const real_t& d, const real_t& r, const real_t& t) {
        const real_t zero(0);
        if (r != zero) {
            const real_t br = b * r;
            if (br != zero)
                return (a + br) * t;
            else
                return a * t + (b * t) * r;
        }
        else
            return (a + d * (b / c)) * t;
    };

    // internal function ladiv1
    auto ladiv1 = [ladiv2](const real_t& a, const real_t& b, const real_t& c,
                           const real_t& d, real_t& p, real_t& q) {
        const real_t r = d / c;
        const real_t t = real_t(1) / (c + d * r);
        p = ladiv2(a, b, c, d, r, t);
        q = ladiv2(b, -a, c, d, r, t);
    };

    // constant to control the lower limit of the overflow threshold
    const real_t bs(2);

    // constants for safe computation
    const real_t one(1);
    const real_t two(2);
    const real_t half(0.5);
    const real_t ov = std::numeric_limits<real_t>::max();
    const real_t safeMin = safe_min<real_t>();
    const real_t u = uroundoff<real_t>();
    const real_t be = bs / (u * u);

    // Treat separate cases of c = 0 and d = 0.
    // It is quicker and prevents the generation of NaNs when doing
    // `d * (b / c)` in ladiv2.
    if (d == real_t(0)) {
        p = a / c;
        q = b / c;
        return;
    }
    if (c == real_t(0)) {
        p = b / d;
        q = -a / d;
        return;
    }

    // constants
    const real_t ab = max(abs(a), abs(b));
    const real_t cd = max(abs(c), abs(d));

    // local variables
    real_t aa = a;
    real_t bb = b;
    real_t cc = c;
    real_t dd = d;
    real_t s(1);  // scaling factor

    // scale values to avoid overflow
    if (ab >= half * ov) {
        aa *= half;
        bb *= half;
        s *= two;
    }
    if (cd >= half * ov) {
        cc *= half;
        dd *= half;
        s *= half;
    }
    if (ab <= safeMin * bs / u) {
        aa *= be;
        bb *= be;
        s /= be;
    }
    if (cd <= safeMin * bs / u) {
        cc *= be;
        dd *= be;
        s *= be;
    }

    // compute the quotient
    if (abs(d) <= abs(c)) {
        ladiv1(aa, bb, cc, dd, p, q);
    }
    else {
        ladiv1(bb, aa, dd, cc, p, q);
        q = -q;
    }

    // scale the result
    if (s != one) {
        p *= s;
        q *= s;
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
template <TLAPACK_COMPLEX T, enable_if_t<is_complex<T>, int> = 0>
T ladiv(const T& x, const T& y)
{
    real_type<T> zr, zi;
    ladiv(real(x), imag(x), real(y), imag(y), zr, zi);

    return T(zr, zi);
}

}  // namespace tlapack

#endif  // TLAPACK_LADIV_HH
