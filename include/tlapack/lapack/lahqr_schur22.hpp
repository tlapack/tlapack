/// @file lahqr_schur22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlanv2.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAHQR_SCHUR22_HH
#define TLAPACK_LAHQR_SCHUR22_HH

#include "tlapack/base/constants.hpp"
#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lapy2.hpp"

namespace tlapack {

/** Computes the Schur factorization of a 2x2 matrix A
 *
 *  A = [a b] = [cs -sn] [aa bb] [ cs sn]
 *      [c d]   [sn  cs] [cc dd] [-sn cs]
 *
 * @param[in,out] a scalar, A(0,0).
 * @param[in,out] b scalar, A(0,1).
 * @param[in,out] c scalar, A(1,0).
 * @param[in,out] d scalar, A(1,1).
 *       On entry, the elements of the matrix A.
 *       On exit, the elements of the Schur factor (aa, bb, cc and dd).
 * @param[out] s1 complex scalar.
 *       First eigenvalue of the matrix.
 * @param[out] s2 complex scalar.
 *       Second eigenvalue of the matrix.
 * @param[out] cs scalar.
 *       Cosine factor of the rotation
 * @param[out] sn scalar.
 *       Sine factor of the rotation
 *
 * @ingroup auxiliary
 */
template <TLAPACK_REAL T>
void lahqr_schur22(T& a,
                   T& b,
                   T& c,
                   T& d,
                   complex_type<T>& s1,
                   complex_type<T>& s2,
                   T& cs,
                   T& sn)
{
    const T zero(0);
    const T half(0.5);
    const T one(1);
    const T multpl(4);

    const T eps = ulp<T>();
    const T safmin = safe_min<T>();
    const T safmn2 = pow(2, T((int)(log2(safmin / eps)) / 2));
    const T safmx2 = one / safmn2;

    if (c == zero) {
        // c is zero, the matrix is already in Schur form.
        cs = one;
        sn = zero;
    }
    else if (b == zero) {
        // b is zero, swapping rows and columns results in Schur form.
        cs = zero;
        sn = one;
        const T temp = d;
        d = a;
        a = temp;
        b = -c;
        c = zero;
    }
    else if ((a - d) == zero and sgn(b) != sgn(c)) {
        cs = one;
        sn = zero;
    }
    else {
        T temp = a - d;
        T p = half * temp;
        const T bcmax = max(abs(b), abs(c));
        const T bcmin = min(abs(b), abs(c)) * T(sgn(b) * sgn(c));
        T scale = max(abs(p), bcmax);
        T z = (p / scale) * p + (bcmax / scale) * bcmin;
        // if z is positive, we should have real eigenvalues
        // however, is z is very small, but positive, we postpone the decision
        if (z >= multpl * eps) {
            // Real eigenvalues.

            // Compute a and d.
            z = p + T(sgn(p)) * sqrt(scale) * sqrt(z);
            a = d + z;
            d = d - (bcmax / z) * bcmin;
            // Compute b and the rotation matrix
            const T tau = lapy2(c, z);
            cs = z / tau;
            sn = c / tau;
            b = b - c;
            c = zero;
        }
        else {
            // Complex eigenvalues, or real (almost) equal eigenvalues.

            // Make diagonal elements equal.
            T sigma = b + c;
            for (int count = 0; count < 20; ++count) {
                scale = max(abs(temp), abs(sigma));
                if (scale >= safmx2) {
                    sigma = sigma * safmn2;
                    temp = temp * safmn2;
                    continue;
                }
                if (scale <= safmn2) {
                    sigma = sigma * safmx2;
                    temp = temp * safmx2;
                    continue;
                }
                break;
            }
            p = half * temp;
            T tau = lapy2(sigma, temp);
            cs = sqrt(half * (one + abs(sigma) / tau));
            sn = -(p / (tau * cs)) * T(sgn(sigma));
            //
            // Compute [aa bb] = [a b][cs -sn]
            //         [cc dd] = [c d][sn  cs]
            //
            const T aa = a * cs + b * sn;
            const T bb = -a * sn + b * cs;
            const T cc = c * cs + d * sn;
            const T dd = -c * sn + d * cs;
            //
            // Compute [a b] = [ cs sn][aa bb]
            //         [c d] = [-sn cs][cc dd]
            //
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;

            temp = half * (a + d);
            a = temp;
            d = temp;

            if (c != zero) {
                if (b != zero) {
                    if (sgn(b) == sgn(c)) {
                        // Real eigenvalues: reduce to upper triangular form
                        const T sab = sqrt(abs(b));
                        const T sac = sqrt(abs(c));
                        p = (c > T(0)) ? sab * sac : -sab * sac;
                        tau = one / sqrt(abs(b + c));
                        a = temp + p;
                        d = temp - p;
                        b = b - c;
                        c = zero;
                        const T cs1 = sab * tau;
                        const T sn1 = sac * tau;
                        temp = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = temp;
                    }
                }
            }
        }
    }
    // Store eigenvalues in s1 and s2
    if (c != zero) {
        const T temp = sqrt(abs(b)) * sqrt(abs(c));
        s1 = complex_type<T>(a, temp);
        s2 = complex_type<T>(d, -temp);
    }
    else {
        s1 = a;
        s2 = d;
    }
}

}  // namespace tlapack

#endif  // TLAPACK_LAHQR_SCHUR22_HH
