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

    if (alphaI == zero) {
        // If alpha is real, then we can use another routine.
        // std::cout << "absI == zero" << std::endl;
        rscl(alphaR, x);
    }

    else if (alphaR == zero) {
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

    else {
        // The following numbers can be computed.
        // They are the inverse of the real and imaginary parts of 1/alpha.
        // Note that a and b are always different from zero.
        // NaNs are only possible if either:
        // 1. alphaR or alphaI is NaN.
        // 2. alphaR and alphaI are both infinite, in which case it makes sense
        // to propagate a NaN.
        real_t a = alphaR + alphaI * (alphaI / alphaR);
        real_t b = alphaI + alphaR * (alphaR / alphaI);

        if (abs(a) < safeMin || abs(b) < safeMin) {
            // This means that both alphaR and alphaI are very small.
            scal(alpha_t(safeMin / a, -safeMin / b), x);
            scal(safeMax, x);
        }
        else if (abs(a) > safeMax || abs(b) > safeMax) {
            if (isinf(alphaR) || isinf(alphaI)) {
                // This means that a and b are both Inf. No need for scaling.
                // Propagates zero.
                scal(alpha_t(one / a, -one / b), x);
            }
            else {
                scal(safeMin, x);
                if (isinf(a) || isinf(b)) {
                    // Infs were generated. We do proper scaling to avoid them.
                    if (absR >= absI) {
                        // |a| <= |b|
                        a = (safeMin * alphaR) +
                            safeMin * (alphaI * (alphaI / alphaR));
                        b = (safeMin * alphaI) +
                            alphaR * ((safeMin * alphaR) / alphaI);
                    }
                    else {
                        // |a| > |b|
                        a = (safeMin * alphaR) +
                            alphaI * ((safeMin * alphaI) / alphaR);
                        b = (safeMin * alphaI) +
                            safeMin * (alphaR * (alphaR / alphaI));
                    }
                    scal(alpha_t(one / a, -one / b), x);
                }
                else {
                    scal(alpha_t(safeMax / a, -safeMax / b), x);
                }
            }
        }
        else {
            // No overflow or underflow.
            scal(alpha_t(one / a, -one / b), x);
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_RSCL_HH
