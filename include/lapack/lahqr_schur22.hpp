/// @file lahqr_schur22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlanv2.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LAHQR_SCHUR22_HH__
#define __TLAPACK_LAHQR_SCHUR22_HH__

#include <complex>
#include <cmath>

#include "base/utils.hpp"
#include "base/types.hpp"

namespace tlapack
{

    /** Computes the Schur factorization of a 2x2 matrix A
     *
     *  A = [a b] = [cs -sn] [aa bb] [ cs sn]
     *      [c d]   [sn  cs] [cc dd] [-sn cs]
     *
     * This routine is designed for real matrices.
     * If the template T is complex, it returns with error
     * and does nothing. (This is so we don't need c++17's static if
     * but still keep the code somewhat clean).
     *
     * @return 0 if the template T is real
     * @return -1 if the template T is complex
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
     * @ingroup geev
     */
    template <
        typename T,
        enable_if_t<!is_complex<T>::value, bool> = true>
    int lahqr_schur22(T &a, T &b, T &c, T &d, std::complex<T> &s1, std::complex<T> &s2, T &cs, T &sn)
    {

        using std::copysign;
        using std::log;
        using std::max;
        using std::min;
        using std::pow;

        const T zero(0);
        const T half(0.5);
        const T one(1);
        const T two(2);
        const T multpl(4);

        const T eps = uroundoff<T>();
        const T safmin = safe_min<T>();
        const T safmn2 = pow(two, (int)(log(safmin / eps) / log(two)) / two);
        const T safmx2 = one / safmn2;

        if (c == zero)
        {
            // c is zero, the matrix is already in Schur form.
            cs = one;
            sn = zero;
        }
        else if (b == zero)
        {
            // b is zero, swapping rows and columns results in Schur form.
            cs = zero;
            sn = one;
            auto temp = d;
            d = a;
            a = temp;
            b = -c;
            c = zero;
        }
        else if ((a - d) == zero and copysign(one, b) != copysign(one, c))
        {
            cs = one;
            sn = zero;
        }
        else
        {
            auto temp = a - d;
            auto p = half * temp;
            auto bcmax = max(abs(b), abs(c));
            auto bcmin = min(abs(b), abs(c)) * copysign(one, b) * copysign(one, c);
            auto scale = max(abs(p), bcmax);
            auto z = (p / scale) * p + (bcmax / scale) * bcmin;
            // if z is positive, we should have real eigenvalues
            // however, is z is very small, but positive, we postpone the decision
            if (z >= multpl * eps)
            {
                // Real eigenvalues.

                // Compute a and d.
                z = p + copysign(one, p) * sqrt(scale) * sqrt(z);
                a = d + z;
                d = d - (bcmax / z) * bcmin;
                // Compute b and the rotation matrix
                auto tau = lapy2(c, z);
                cs = z / tau;
                sn = c / tau;
                b = b - c;
                c = zero;
            }
            else
            {
                // Complex eigenvalues, or real (almost) equal eigenvalues.

                // Make diagonal elements equal.
                auto sigma = b + c;
                for (int count = 0; count < 20; ++count)
                {
                    scale = max(abs(temp), abs(sigma));
                    if (scale >= safmx2)
                    {
                        sigma = sigma * safmn2;
                        temp = temp * safmn2;
                        continue;
                    }
                    if (scale <= safmn2)
                    {
                        sigma = sigma * safmx2;
                        temp = temp * safmx2;
                        continue;
                    }
                    break;
                }
                p = half * temp;
                auto tau = lapy2(sigma, temp);
                cs = sqrt(half * (one + abs(sigma) / tau));
                sn = -(p / (tau * cs)) * copysign(one, sigma);
                //
                // Compute [aa bb] = [a b][cs -sn]
                //         [cc dd] = [c d][sn  cs]
                //
                auto aa = a * cs + b * sn;
                auto bb = -a * sn + b * cs;
                auto cc = c * cs + d * sn;
                auto dd = -c * sn + d * cs;
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

                if (c != zero)
                {
                    if (b != zero)
                    {
                        if (copysign(one, b) == copysign(one, c))
                        {
                            // Real eigenvalues: reduce to upper triangular form
                            auto sab = sqrt(abs(b));
                            auto sac = sqrt(abs(c));
                            p = copysign(sab * sac, c);
                            tau = one / sqrt(abs(b + c));
                            a = temp + p;
                            d = temp - p;
                            b = b - c;
                            c = zero;
                            auto cs1 = sab * tau;
                            auto sn1 = sac * tau;
                            temp = cs * cs1 - sn * sn1;
                            sn = cs * sn1 + sn * cs1;
                            cs = temp;
                        }
                    }
                }
            }
        }
        // Store eigenvalues in s1 and s2
        if (c != zero)
        {
            auto temp = sqrt(abs(b)) * sqrt(abs(c));
            s1 = std::complex<T>(a, temp);
            s2 = std::complex<T>(d, -temp);
        }
        else
        {
            s1 = a;
            s2 = d;
        }
        return 0;
    }

    template <
        typename T,
        enable_if_t<is_complex<T>::value, bool> = true>
    int lahqr_schur22(T &a, T &b, T &c, T &d, T &s1, T &s2, real_type<T> &cs, T &sn)
    {
        return -1;
    }

} // lapack

#endif // __LAHQR_SCHUR22_HH__
