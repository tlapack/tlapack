/// @file lahqz_eig22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlag2.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LAHQZ_EIG22_HH__
#define __TLAPACK_LAHQZ_EIG22_HH__

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lahqr_eig22.hpp"

namespace tlapack {

/** Computes the generalized eigenvalues of a 2x2 pencil (A,B) with B upper
 * triangular
 *
 * @param[in] A 2x2 matrix
 * @param[in] B 2x2 upper triangular matrix
 * @param[out] alpha1 complex number
 * @param[out] alpha2 complex number
 * @param[out] beta1 number
 * @param[out] beta2 number
 *                On exit, (alpha1, beta1), (alpha2, beta2) are the generalized
 *                eigenvalues of the pencil (A,B)
 *
 */
template <TLAPACK_MATRIX A_t, TLAPACK_MATRIX B_t, TLAPACK_SCALAR T>
void lahqz_eig22(const A_t& A,
                 const B_t& B,
                 complex_type<T>& alpha1,
                 complex_type<T>& alpha2,
                 T& beta1,
                 T& beta2)
{
    // Using
    using TA = type_t<A_t>;
    using real_t = real_type<TA>;

    // Constants
    const real_t zero(0);
    const real_t half(0.5);
    const real_t one(1);
    const real_t two(2);
    const real_t safmin = safe_min<real_t>();
    const real_t rtmin = sqrt(safmin);
    const real_t rtmax = sqrt(safe_max<real_t>());
    const real_t safmax = one / safmin;

    //
    // Scale A
    //
    real_t anorm =
        std::max<real_t>(std::max<real_t>(abs1(A(0, 0)) + abs1(A(1, 0)),
                                          abs1(A(0, 1)) + abs1(A(1, 1))),
                         safmin);
    real_t ascale = one / anorm;
    TA a00 = ascale * A(0, 0);
    TA a01 = ascale * A(0, 1);
    TA a10 = ascale * A(1, 0);
    TA a11 = ascale * A(1, 1);
    //
    // If B is singular, deflate infinite eigenvalue
    // Note, here we deviate from LAPACK, as LAPACK perturbs B
    // to make it non-singular, but this removes the possibility
    // for infinite eigenvalues
    //
    TA b00 = B(0, 0);
    TA b01 = B(0, 1);
    TA b11 = B(1, 1);
    real_t bmin =
        rtmin * std::max<real_t>(std::max<real_t>(abs1(b00), abs1(b01)),
                                 std::max<real_t>(abs1(b11), rtmin));
    if (abs1(b00) < bmin) {
        b00 = zero;
        // B(0,0) is zero, we can apply rotations to the left
        // to make A(1,0) zero without perturbing the upper triangular
        // structure of B
        if (a10 != zero) {
            real_t c;
            TA s;
            rotg(a00, a10, c, s);
            a10 = zero;
            TA temp = c * a01 + s * a11;
            a11 = c * a11 - conj(s) * a01;
            a01 = temp;
            temp = c * b01 + s * b11;
            b11 = c * b11 - conj(s) * b01;
            b01 = temp;
        }

        // Apply scaling to the right of B to make B(1,1) real
        if constexpr (is_complex<TA>) {
            if (abs1(b11) < bmin) {
                b11 = zero;
            }
            else {
                TA scale = conj(b11) / abs(b11);
                b01 *= scale;
                b11 = abs(b11);
                a01 *= scale;
                a11 *= scale;
            }
        }

        alpha1 = a00;
        alpha2 = a11;
        beta1 = zero;
        beta2 = real(b11) * ascale;
        return;
    }

    if (abs1(b11) < bmin) {
        b11 = zero;
        // B(1,1) is zero, we can apply rotations to the right
        // to make A(1,0) zero without perturbing the upper triangular
        // structure of B
        if (a10 != zero) {
            real_t c;
            TA s;
            rotg(a11, a10, c, s);
            a10 = zero;
            TA temp = c * a01 + s * a00;
            a00 = c * a00 - conj(s) * a01;
            a01 = temp;
            temp = c * b01 + s * b00;
            b00 = c * b00 - conj(s) * b01;
            b01 = temp;
        }

        // Apply scaling to the right of B to make B(0,0) real
        if constexpr (is_complex<TA>) {
            if (abs1(b00) < bmin) {
                b00 = zero;
            }
            else {
                TA scale = conj(b00) / abs(b00);
                b01 *= scale;
                b00 = abs(b00);
                a01 *= scale;
                a00 *= scale;
            }
        }

        alpha1 = a00;
        alpha2 = a11;
        beta1 = real(b00) * ascale;
        beta2 = zero;
        return;
    }
    //
    // Scale B
    //
    real_t bnorm = std::max<real_t>(
        std::max<real_t>(abs1(b00), abs1(b01) + abs1(b11)), safmin);
    real_t bsize = std::max<real_t>(abs1(b00), abs1(b11));
    real_t bscale = one / bsize;
    b00 = bscale * b00;
    b01 = bscale * b01;
    b11 = bscale * b11;
    //
    // Compute larger eigenvalue by method described by C. van Loan
    // ( AS is A shifted by -SHIFT*B )
    // TODO: add specific reference, van Loan wrote a lot of things
    //
    TA binv00 = one / b00;
    TA binv11 = one / b11;
    TA s0 = a00 * binv00;
    TA s1 = a11 * binv11;
    TA as00, as01, as11, ss, abi11, pp, shift;
    if (abs1(s0) <= abs1(s1)) {
        as01 = a01 - s0 * b01;
        as11 = a11 - s0 * b11;
        ss = a10 * (binv00 * binv11);
        abi11 = as11 * binv11 - ss * b01;
        pp = half * abi11;
        shift = s0;
    }
    else {
        as01 = a01 - s1 * b01;
        as00 = a00 - s1 * b00;
        ss = a10 * (binv00 * binv11);
        abi11 = -ss * b01;
        pp = half * (as00 * binv00 + abi11);
        shift = s1;
    }
    TA qq = ss * as01;
    TA discr;
    real_t r;
    if (abs1(pp * rtmin) >= one) {
        discr = (rtmin * pp) * (rtmin * pp) + qq * safmin;
        r = sqrt(abs(discr)) * rtmax;
    }
    else {
        if (abs1(pp * pp) + abs1(qq) <= safmin) {
            discr = (rtmax * pp) * (rtmax * pp) + qq * safmax;
            r = sqrt(abs(discr)) * rtmin;
        }
        else {
            discr = pp * pp + qq;
            r = sqrt(abs(discr));
        }
    }

    if constexpr (is_complex<TA>) {
        TA root = sqrt(discr);

        TA mu_big;
        if (abs1(pp + root) > abs1(pp - root)) {
            mu_big = pp + root;
        }
        else {
            mu_big = pp - root;
        }
        TA mu_small = -qq / mu_big;
        alpha1 = shift + mu_big;
        alpha2 = shift + mu_small;
    }
    else {
        //
        // Real pencil, check if we have 2 real eigenvalues
        // or a complex conjugate pair
        //
        if (discr >= zero or r == zero) {
            // real eigenvalues
            TA rpp = pp > zero ? r : -r;
            TA sum = pp + rpp;
            TA diff = pp - rpp;
            TA wbig = shift + sum;
            //
            // Compute smaller eigenvalue
            //
            TA wsmall = shift + diff;
            if (half * abs1(wbig) > max<real_t>(abs1(wsmall), safmin)) {
                T wdet = (a00 * a11 - a01 * a10) * (binv00 * binv11);
                wsmall = wdet / wbig;
            }
            //
            // Choose (real) eigenvalue closest to 1,1 element of AB^{-1}
            // For alpha1
            //
            if (pp > abi11) {
                alpha1 = min<real_t>(wbig, wsmall);
                alpha2 = max<real_t>(wbig, wsmall);
            }
            else {
                alpha1 = max<real_t>(wbig, wsmall);
                alpha2 = min<real_t>(wbig, wsmall);
            }
        }
        else {
            // complex conjugate eigenvalues
            alpha1 = complex_type<TA>(shift + pp, r);
            alpha2 = complex_type<TA>(shift + pp, -r);
        }
    }
    //
    // Further scaling to avoid underflow and overflow in computing
    // beta1 and overflow in computing w*B.
    //
    // This scale factor (WSCALE) is bounded from above using C1 and C2,
    // and from below using C3 and C4.
    //    C1 implements the condition  s A  must never overflow.
    //    C2 implements the condition  w B  must never overflow.
    //    C3, with C2,
    //       implement the condition that s A - w B must never overflow.
    //    C4 implements the condition  s    should not underflow.
    //    C5 implements the condition  max<real_t>(s,|w|) should be at least 2.
    real_t c1 = bsize * (safmin * max<real_t>(one, ascale));
    real_t c2 = safmin * max<real_t>(one, bnorm);
    real_t c3 = bsize * safmin;
    real_t c4;
    if (ascale <= one and bsize <= one) {
        c4 = min<real_t>(one, (ascale / safmin) * bsize);
    }
    else {
        c4 = one;
    }
    real_t c5;
    if (ascale <= one or bsize <= one) {
        c5 = min<real_t>(one, ascale * bsize);
    }
    else {
        c5 = one;
    }
    //
    // Scale first eigenvalue
    //
    real_t wabs = abs1(alpha1);
    real_t fuzzy1 = one + real_t(1.0e-5);
    real_t wsize =
        max<real_t>(max<real_t>(safmin, c1),
                    max<real_t>(fuzzy1 * (wabs * c2 + c3),
                                min<real_t>(c4, half * max<real_t>(wabs, c5))));
    if (wsize != one) {
        real_t wscale = one / wsize;
        if (wsize > one) {
            beta1 = (max<real_t>(ascale, bsize) * wscale) *
                    min<real_t>(ascale, bsize);
        }
        else {
            beta1 = (min<real_t>(ascale, bsize) * wscale) *
                    max<real_t>(ascale, bsize);
        }
        alpha1 = wscale * alpha1;
    }
    else {
        beta1 = ascale * bsize;
        beta2 = beta1;
    }

    wabs = abs1(alpha2);
    wsize =
        max<real_t>(max<real_t>(safmin, c1),
                    max<real_t>(fuzzy1 * (wabs * c2 + c3),
                                min<real_t>(c4, half * max<real_t>(wabs, c5))));
    if (wsize != one) {
        real_t wscale = one / wsize;
        if (wsize > one) {
            beta2 = (max<real_t>(ascale, bsize) * wscale) *
                    min<real_t>(ascale, bsize);
        }
        else {
            beta2 = (min<real_t>(ascale, bsize) * wscale) *
                    max<real_t>(ascale, bsize);
        }
        alpha2 = wscale * alpha2;
    }
    else {
        beta2 = ascale * bsize;
    }
}

}  // namespace tlapack

#endif  // __LAHQZ_EIG22_HH__
