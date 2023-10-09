/// @file svd22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlasv2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_SVD22_HH
#define TLAPACK_SVD22_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lapy2.hpp"

namespace tlapack {

/** Computes the singular value decomposition of a 2-by-2
 * real triangular matrix
 *
 *     T = [  F   G  ]
 *         [  0   H  ].
 *
 *  On return, abs(SSMAX) is the larger singular value, abs(SSMIN) is the
 *  smaller singular value, and (CSL,SNL) and (CSR,SNR) are the left and
 *  right singular vectors for abs(SSMAX), giving the decomposition
 *
 *     [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
 *     [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ].
 *
 * @param[in] f scalar, T(0,0).
 * @param[in] g scalar, T(0,1).
 * @param[in] h scalar, T(1,1).
 * @param[out] ssmin scalar.
 *       abs(ssmin) is the smaller singular value.
 * @param[out] ssmax scalar.
 *       abs(ssmax) is the larger singular value.
 * @param[out] csl scalar.
 *       Cosine factor of the rotation from the left.
 * @param[out] snl scalar.
 *       Sine factor of the rotation from the left.
 *       The vector (CSL, SNL) is a unit left singular vector for the
 *       singular value abs(SSMAX).
 * @param[out] csr scalar.
 *       Cosine factor of the rotation from the right.
 * @param[out] snr scalar.
 *       Sine factor of the rotation from the right.
 *       The vector (CSR, SNR) is a unit right singular vector for the
 *       singular value abs(SSMAX).
 *
 * @ingroup auxiliary
 */
template <typename T, enable_if_t<!is_complex<T>, bool> = true>
void svd22(const T& f,
           const T& g,
           const T& h,
           T& ssmin,
           T& ssmax,
           T& csl,
           T& snl,
           T& csr,
           T& snr)
{
    const T eps = ulp<T>();
    const T zero(0);
    const T one(1);
    const T two(2);
    const T half(0.5);
    const T four(4);

    T ft = f;
    T fa = abs(ft);
    T ht = h;
    T ha = abs(h);

    // PMAX points to the maximum absolute element of matrix
    // PMAX = 1 if F largest in absolute values
    // PMAX = 2 if G largest in absolute values
    // PMAX = 3 if H largest in absolute values
    int pmax = 1;

    bool swap = (ha > fa);
    if (swap) {
        pmax = 3;
        auto temp = ft;
        ft = ht;
        ht = temp;
        temp = fa;
        fa = ha;
        ha = temp;
        // Now FA > HA
    }
    T gt = g;
    T ga = abs(gt);

    T clt, crt, slt, srt;
    if (ga == zero) {
        // Diagonal matrix
        ssmin = ha;
        ssmax = fa;
        clt = one;
        crt = one;
        slt = zero;
        srt = zero;
    }
    else {
        bool gasmal = true;
        if (ga > fa) {
            pmax = 2;
            if ((fa / ga) < eps) {
                // ga is very large
                gasmal = false;
                ssmax = ga;
                if (ha > one) {
                    ssmin = fa / (ga / ha);
                }
                else {
                    ssmin = (fa / ga) * ha;
                }
                clt = one;
                slt = ht / gt;
                srt = one;
                crt = ft / gt;
            }
        }
        if (gasmal) {
            // Normal case
            T d, l;
            d = fa - ha;
            // Note that 0 < l < 1
            if (d == fa) {
                l = one;
            }
            else {
                l = d / fa;
            }
            // Note that abs(m) < 1/eps
            T m = gt / ft;
            // Note that t >= 1
            T t = two - l;
            T mm = m * m;
            T tt = t * t;
            // Note that 1 <= s <= 1 + 1/eps
            T s = sqrt(tt + mm);
            // Note that 0 <= r <= 1 + 1/eps
            T r;
            if (l == zero) {
                r = abs(m);
            }
            else {
                r = sqrt(l * l + mm);
            }
            // Note that 1 <= a <= 1 + abs(m)
            T a = half * (s + r);
            ssmin = ha / a;
            ssmax = fa * a;
            if (mm == zero) {
                // m is very tiny, so m*m has underflowed
                if (l == zero) {
                    t = two * T(sgn(ft)) * T(sgn(gt));
                }
                else {
                    t = gt / (abs(d) * T(sgn(ft))) + m / t;
                }
            }
            else {
                t = (m / (s + t) + m / (r + l)) * (one + a);
            }
            l = sqrt(t * t + four);
            crt = two / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    if (swap) {
        csl = srt;
        snl = crt;
        csr = slt;
        snr = clt;
    }
    else {
        csl = clt;
        snl = slt;
        csr = crt;
        snr = srt;
    }
    //
    // Correct signs of SSMAX and SSMIN
    //
    T tsign;
    if (pmax == 1)
        tsign = T(sgn(csr)) * T(sgn(csl)) * T(sgn(f));
    else if (pmax == 2)
        tsign = T(sgn(snr)) * T(sgn(csl)) * T(sgn(g));
    else
        tsign = T(sgn(snr)) * T(sgn(snl)) * T(sgn(h));
    ssmax = ssmax * T(sgn(tsign));
    ssmin = ssmin * T(sgn(tsign * T(sgn(f)) * T(sgn(h))));
}

}  // namespace tlapack

#endif  // TLAPACK_SVD22_HH
