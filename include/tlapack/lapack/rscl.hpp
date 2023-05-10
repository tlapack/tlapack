/// @file rscl.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_RSCL_HH
#define TLAPACK_RSCL_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/ladiv.hpp"

namespace tlapack {

template <class vector_t,
          class alpha_t,
          enable_if_t<!is_complex<alpha_t>::value, int> = 0>
void rscl(const alpha_t& alpha, vector_t& x)
{
    using real_t = real_type<alpha_t>;

    // constants
    const real_t safeMax = safe_max<real_t>();
    const real_t safeMin = safe_min<real_t>();
    const real_t r = abs(alpha);

    if (r > safeMax) {
        scal(safeMin, x);
        scal(safeMax / alpha, x);
    }
    else if (r < safeMin) {
        scal(safeMin / alpha, x);
        scal(safeMax, x);
    }
    else
        scal(real_t(1) / alpha, x);
}

/**
 * Scale vector by the reciprocal of a constant, $x := x / \alpha$.
 *
 * If alpha is real, then this routine is equivalent to scal(1/alpha, x). This
 * is done without overflow or underflow as long as the final result x/a does
 * not overflow or underflow.
 *
 * If alpha is complex, then we use the following algorithm:
 * 1. If the real part of alpha is zero, then scale by the reciprocal of the
 * imaginary part of alpha.
 * 2. If either real or imaginary part is greater than or equal to safeMax. If
 * so, do proper scaling using a reliable algorithm for complex division.
 * 3. If 1 and 2 are false, we can compute the reciprocal of real and imaginary
 * parts of 1/alpha without NaNs. If both components are in the safe range, then
 * we can do the reciprocal without overflow or underflow. Otherwise, we do
 * proper scaling.
 *
 * @note For complex alpha, it is important to always scale first by the
 * smallest factor. This avoids an early overflow that may lead to a NaN.
 * For example, if alpha = 5.877472e-39 + 2.802597e-45 * I and v[i] = 30 + 30 *
 * I in single precision, we want to scale v[i] first by (safeMin / alpha) and
 * after that scale the result by safeMax. Doing it the other way around will
 * possibly lead to a (Inf - Inf) which should be tagged as a NaN.
 *
 * @param[in] alpha Scalar.
 * @param[in,out] x A n-element vector.
 *
 * @ingroup auxiliary
 */
template <class vector_t,
          class alpha_t,
          enable_if_t<is_complex<alpha_t>::value, int> = 0>
void rscl(const alpha_t& alpha, vector_t& x)
{
    using real_t = real_type<alpha_t>;

    const real_t safeMax = safe_max<real_t>();
    const real_t safeMin = safe_min<real_t>();
    const real_t zero(0);
    const real_t one(1);

    const real_t alphaR = real(alpha);
    const real_t alphaI = imag(alpha);
    const real_t absR = abs(alphaR);
    const real_t absI = abs(alphaI);

    if (absI == zero) {
        // If alpha is real, then we can use another routine.
        // std::cout << "absI == zero" << std::endl;
        rscl(alphaR, x);
    }

    else if (absR == zero) {
        // If alpha has a zero real part, then we follow the same rules as if
        // alpha were real.
        if (absI > safeMax) {
            scal(safeMin, x);
            scal(alpha_t(zero, -safeMax / alphaI), x);
        }
        else if (absI < safeMin) {
            scal(alpha_t(zero, -safeMin / alphaI), x);
            scal(safeMax, x);
        }
        else
            scal(alpha_t(zero, -one / alphaI), x);
    }

    else if (absR >= safeMax || absI >= safeMax) {
        // Either real or imaginary part is too large.
        const alpha_t invAlpha = ladiv(alpha_t(safeMax), alpha);
        scal(safeMin, x);
        scal(invAlpha, x);
    }

    else {
        // The following numbers can be computed without NaNs and zeros.
        // They do not overflow simultaneously.
        // They are the inverse of the real and imaginary parts of 1/alpha.
        const real_t a = alphaR + alphaI * (alphaI / alphaR);
        const real_t b = alphaI + alphaR * (alphaR / alphaI);

        if (abs(a) < safeMin || abs(b) < safeMin) {
            const alpha_t invAlpha(safeMin / a, -safeMin / b);
            scal(invAlpha, x);
            scal(safeMax, x);
        }
        else if (isinf(a)) {
            const real_t aScaled =
                (absR >= absI) ? (safeMin * alphaR) +
                                     alphaI * (safeMin * (alphaI / alphaR))
                               : (safeMin * alphaR) +
                                     alphaI * ((safeMin * alphaI) / alphaR);
            const alpha_t invAlpha(one / aScaled, -safeMax / b);
            scal(safeMin, x);
            scal(invAlpha, x);
        }
        else if (isinf(b)) {
            const real_t bScaled =
                (absI >= absR) ? (safeMin * alphaI) +
                                     alphaR * (safeMin * (alphaR / alphaI))
                               : (safeMin * alphaI) +
                                     alphaR * ((safeMin * alphaR) / alphaI);
            const alpha_t invAlpha(safeMax / a, -one / bScaled);
            scal(safeMin, x);
            scal(invAlpha, x);
        }
        else if (abs(a) > safeMax || abs(b) > safeMax) {
            const alpha_t invAlpha(safeMax / a, -safeMax / b);
            scal(safeMin, x);
            scal(invAlpha, x);
        }
        else {
            const alpha_t invAlpha(one / a, -one / b);
            scal(invAlpha, x);
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_RSCL_HH
