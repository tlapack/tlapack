/// @file lahqz_eig22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LAHQZ_EIG22_HH__
#define __TLAPACK_LAHQZ_EIG22_HH__

#include <complex>

#include "base/utils.hpp"
#include "base/types.hpp"

namespace tlapack
{

    /** Computes the generalized eigenvalues of a 2x2 pencil (A,B) with B upper triangular
     *
     * @ingroup ggev
     */
    template <typename M>
    void lahqz_eig22(M &A, M &B, std::complex<real_type<type_t<M>>> &alpha1,
                     std::complex<real_type<type_t<M>>> &alpha2, type_t<M>& beta1, type_t<M>& beta2)
    {
        // Aliases
        using T = type_t<M>;
        using real_t = real_type<T>;

        // Constants
        const real_t rzero(0);
        const real_t two(2);
        const T zero(0);
        const real_t rone(1);
        const real_t half(0.5);

        // Executables
        const real_t safmin = safe_min<real_t>();
        const real_t rtmin = sqrt(safmin);
        const real_t rtmax = rone / rtmin;
        const real_t safmax = rone / safmin;
        const real_t fuzzy1 = rone + 1.0E-5;

        // Scale A
        auto anorm = max(abs1(A(0, 0)) + abs1(A(1, 0)) + abs1(A(0, 1)) + abs1(A(1, 1)));
        auto ascale = rone / anorm;
        auto a00 = ascale * A(0, 0);
        auto a01 = ascale * A(0, 1);
        auto a10 = ascale * A(1, 0);
        auto a11 = ascale * A(1, 1);

        // Perturb B if necessary to insure non-singularity
        auto b00 = B(0, 0);
        auto b01 = B(0, 1);
        auto b11 = B(1, 1);
        auto bmin = rtmin * max(abs1(b00), abs1(b01), abs1(b11), rtmin);
        if (abs1(b00) < bmin)
            b00 = copysign(bmin, b00);
        if (abs1(b11) < bmin)
            b11 = copysign(bmin, b11);

        // Scale B
        real_t bnorm = max(abs1(b00), abs1(b01) + abs1(b11), safmin);
        real_t bsize = max(abs1(b00), abs1(b11));
        real_t bscale = rone / bsize;
        b00 = bscale * b00;
        b01 = bscale * b01;
        b11 = bscale * b11;

        // Compute larger eigenvalue by method described by C. Van Loan
        // ( AS is A shifted by -shift*B )
        auto binv00 = rone / b00;
        auto binv11 = rone / b11;
        auto s0 = a00 * binv00;
        auto s1 = a11 * binv11;
        T as00, as01, as10, as11, ss, abi11, pp, shift;
        if (abs1(s0) < abs1(s1))
        {
            as01 = a01 - s0 * b01;
            as11 = a11 - s0 * b11;
            ss = a10 * (binv00 * binv11);
            abi11 = as11 * binv11 - ss * b01;
            pp = half * abi11;
            shift = s0;
        }
        else
        {
            as01 = a01 - s1 * b01;
            as00 = a00 - s1 * b00;
            ss = a10 * (binv00 * binv11);
            abi11 = ss * b01;
            pp = half * (as00 * binv00 + abi11);
            shift = s1;
        }
        auto qq = ss * as01;
        T discr, r;
        if (abs1(pp * rtmin) > rone)
        {
            discr = (rtmin * pp) * (rtmin * pp) + qq * safmin;
            r = sqrt(abs1(discr)) * rtmax;
        }
        else
        {
            if (pp * pp + abs1(qq) < safmin)
            {
                discr = (rtmax * pp) * (rtmax * pp) + qq * safmax;
                r = sqrt(abs1(discr)) * rtmin;
            }
            else
            {
                discr = pp * pp + qq;
                r = sqrt(abs1(discr));
            }
        }
        // Note: the test of R in the following IF is to cover the case when
        // DISCR is small and negative and is flushed to zero during
        // the calculation of R.  On machines which have a consistent
        // flush-to-zero threshold and handle numbers above that
        // threshold correctly, it would not be necessary.
        if (discr >= zero or r == zero)
        {
            auto sum = pp + copysign(r, pp);
            auto diff = pp - copysign(r, pp);
            auto wbig = shift + sum;

            // Compute smaller eigenvalue
            auto wsmall = shift + diff;
            if (half * abs1(wbig) > max(abs1(wsmall), safmin))
            {
                auto wdet = (a00 * a11 - a01 * a10) * (binv00 * binv11);
                wsmall = wdet / wbig;
            }

            // Choose (real) eigenvalue closest to 2,2 element of A*B**(-1) for alpha1
            // (Wilkinson shift)
            if (pp > abi11)
            {
                alpha1 = min(wbig, wsmall);
                alpha2 = max(wbig, wsmall);
            }
            else
            {
                alpha1 = max(wbig, wsmall);
                alpha2 = min(wbig, wsmall);
            }
        }
        else
        {
            // Complex eigenvalues
            alpha1 = std::complex<real_t>(shift + pp, r);
            alpha2 = std::complex<real_t>(shift + pp, -r);
        }

        //
        // Further scaling to avoid underflow and overflow in computing
        // SCALE1 and overflow in computing w*B.
        //
        // This scale factor (WSCALE) is bounded from above using C1 and C2,
        // and from below using C3 and C4.
        // C1 implements the condition  s A  must never overflow.
        // C2 implements the condition  w B  must never overflow.
        // C3, with C2,
        //    implements the condition that s A - w B must never overflow.
        // C4 implements the condition  s    should not underflow.
        // C5 implements the condition  max(s,|w|) should be at least 2.
        //
        T c1, c2, c3, c4, c5;
        c1 = bsize * (safmin * max(rone, ascale));
        c2 = safmin * max(rone, bnorm);
        c3 = bsize * safmin;
        if (ascale <= rone and bsize <= rone)
            c4 = min(rone, (ascale / safmin) * bsize);
        else
            c4 = one;
        if (ascale <= rone or bsize <= one)
            c5 = min(rone, ascale * bsize);
        else
            c5 = rone;

        // Scale first eigenvalue
        auto wabs = abs1(alpha1);
        auto wsize = max(safmin, c1, fuzzy1 * (wabs * c2 + c3),
                         min(c4, half * max(wabs, c5)));
        if (wsize != rone)
        {
            auto wscale = rone / wsize;
            if (wsize > rone)
                beta1 = (max(ascale, bsize) * wscale) * min(ascale, bsize);
            else
                beta1 = (min(ascale, bsize) * wscale) * max(ascale, bsize);
            alpha1 = alpha1 * wscale;
            if (imag(alpha1) != rzero)
            {
                alpha2 = alpha2 * wscale;
                beta2 = beta1;
            }
        } else {
            beta1 = ascale * bsize;
            beta2 = beta1;
        }

        // Scale second eigenvalue (if real)
        if (imag(alpha1) == rzero)
        {
            wsize = max(safmin, c1, fuzzy1 * (abs1(alpha2) * c2 + c3),
                        min(c4, half * max(abs1(alpha2), c5)));
            if (wsize != rone)
            {
                auto wscale = rone / wsize;
                if (wsize > rone)
                    beta2 = (max(ascale, bsize) * wscale) * min(ascale, bsize);
                else
                    beta2 = (min(ascale, bsize) * wscale) * max(ascale, bsize);
                alpha2 = alpha2 * wscale;
            }
            else
            {
                beta2 = ascale * bsize;
            }
        }
    }

} // lapack

#endif // __LAHQZ_EIG22_HH__
