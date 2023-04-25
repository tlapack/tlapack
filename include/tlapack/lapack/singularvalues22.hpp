/// @file singularvalues22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlasv2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_SINGULARVALUES22_HH
#define TLAPACK_SINGULARVALUES22_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lapy2.hpp"

namespace tlapack {

/** Computes the singular value decomposition of a 2-by-2
 * real triangular matrix
 *
 *     T = [  F   G  ]
 *         [  0   H  ].
 *
 *  On return, SSMAX is the larger singular value and SSMIN is the
 *  smaller singular value.
 *
 * @param[in] f scalar, T(0,0).
 * @param[in] g scalar, T(0,1).
 * @param[in] h scalar, T(1,1).
 * @param[out] ssmin scalar.
 *       ssmin is the smaller singular value.
 * @param[out] ssmax scalar.
 *       ssmax is the larger singular value.
 *
 * @ingroup auxiliary
 */
template <typename T, enable_if_t<!is_complex<T>::value, bool> = true>
void singularvalues22(const T& f, const T& g, const T& h, T& ssmin, T& ssmax)
{
    const T zero(0);
    const T one(1);
    const T two(2);

    T fa = abs(f);
    T ga = abs(g);
    T ha = abs(h);
    T fhmn = min(fa, ha);
    T fhmx = max(fa, ha);
    if (fhmn == zero) {
        ssmin = zero;
        if (fhmx == zero)
            ssmax = ga;
        else
            ssmax = max(fhmx, ga) *
                    sqrt(one + square(min(fhmx, ga) / max(fhmx, ga)));
    }
    else {
        T as, at, au, c;
        if (ga < fhmx) {
            as = one + fhmn / fhmx;
            at = (fhmx - fhmn) / fhmx;
            au = square(ga / fhmx);
            c = two / (sqrt(as * as + au) + sqrt(at * at + au));
            ssmin = fhmn * c;
            ssmax = fhmx / c;
        }
        else {
            au = fhmx / ga;
            if (au == zero) {
                //   Avoid possible harmful underflow if exponent range
                //   asymmetric (true ssmin may not underflow even if
                //   au underflows)
                ssmin = (fhmn * fhmx) / ga;
                ssmax = ga;
            }
            else {
                as = one + fhmn / fhmx;
                at = (fhmx - fhmn) / fhmx;
                c = one /
                    (sqrt(one + square(as * au)) + sqrt(one + square(at * au)));
                ssmin = (fhmn * c) * au;
                ssmin = ssmin + ssmin;
                ssmax = ga / (c + c);
            }
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_SINGULARVALUES22_HH
