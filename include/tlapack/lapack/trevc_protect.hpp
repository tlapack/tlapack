/// @file trevc_protect.hpp
/// @author Thijs Steel, KU Leuven, Belgium
// based on A. Schwarz et al., "Robust parallel eigenvector computation for the
// non-symmetric eigenvalue problem"
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC_PROTECT_HH
#define TLAPACK_TREVC_PROTECT_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"

namespace tlapack {

/**
 * Given two numbers a and b, calculate a scaling factor alpha \in (0, 1] such
 * that the division (a * alpha) / (b) cannot be larger than the threshold
 * safe_max
 *
 * See algorithm 2 in "Robust solution of triangular linear systems"
 *
 * @param a Numerator
 * @param b Denominator
 * @param sf_min Safe minimum threshold (1/sf_max)
 * @param sf_max Safe maximum threshold
 *
 * @return Scaling factor alpha
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_real<T>, int> = 0>
T trevc_protectdiv(T a, T b, T sf_min, T sf_max)
{
    T alpha = T(1);

    if (abs(b) < sf_min) {
        if (abs(a) > abs(b) * sf_max) alpha = (abs(b) * sf_max) / abs(a);
    }
    else {
        if (abs(b) < T(1))
            if (abs(a) > abs(b) * sf_max) alpha = T(1) / abs(a);
    }
    return alpha;
}

/**
 * Given two numbers a and b, calculate a scaling factor alpha \in (0, 1] such
 * that the division (a * alpha) / (b) cannot be larger than the threshold
 * safe_max
 *
 * This is the version for complex numbers which was not described in the paper
 * The version described in the paper should actually work when using abs, but
 * here we use abs1, which requires some extra care.
 *
 * We use the property that abs(z) <= abs1(z) <= sqrt(2) * abs(z) < 2*abs(z)
 * to transform alpha * abs(a) <= safe_max * abs(b)
 * into 2 * alpha * abs1(a) <= safe_max * abs1(b)
 * which transforms the problem into the real-valued version.
 *
 * @param a Numerator
 * @param b Denominator
 * @param sf_min Safe minimum threshold (1/sf_max)
 * @param sf_max Safe maximum threshold
 *
 * @return Scaling factor alpha
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_complex<T>, int> = 0>
real_type<T> trevc_protectdiv(T a,
                              T b,
                              real_type<T> sf_min,
                              real_type<T> sf_max)
{
    real_type<T> a1 = real_type<T>(2) * abs1(a);
    real_type<T> b1 = abs1(b);

    return trevc_protectdiv(a1, b1, sf_min, sf_max);
}

/**
 * Given the infinity norms of the matrices Y, T, and X, calculate a scaling
 * factor xi \in (0, 1] such that the update Z = xi*Y - T * (xi * X) does not
 * overflow
 *
 * See algorithm 3 in "Robust solution of triangular linear systems"
 *
 * We note that this algorithm should also work if the infinity norm of
 * a complex type is calculated using abs1 instead of abs.
 *
 * @param ynorm Infinity norm of Y
 * @param tnorm Infinity norm of T
 * @param xnorm Infinity norm of X
 * @param sf_max Safe maximum threshold
 *
 * @return Scaling factor alpha
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_real<T>, int> = 0>
T trevc_protectupdate(T ynorm, T tnorm, T xnorm, T sf_max)
{
    T xi = T(1);
    if (xnorm <= T(1)) {
        if (tnorm * xnorm > sf_max - ynorm) {
            xi = T(0.5);
        }
    }
    else {
        if (tnorm > (sf_max - ynorm) / xnorm) {
            xi = T(1) / (T(2) * xnorm);
        }
    }
    return xi;
}

/**
 * Calculate a scaling factor xi \in [0.5, 1] such that the sum (xi * a) + (xi *
 * b) does not overflow
 *
 * @param a First addend
 * @param b Second addend
 *
 * @return Scaling factor
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_real<T>, int> = 0>
T trevc_protectsum(T a, T b, T sf_max)
{
    T xi = T(1);
    if (a * b > 0)
        if (abs(a) > (sf_max - abs(b))) xi = T(0.5);
    return xi;
}

/**
 * Calculate a scaling factor xi \in [0.5, 1] such that the sum (xi * a) + (xi *
 * b) does not overflow
 *
 * @param a First addend
 * @param b Second addend
 *
 * @return Scaling factor
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_complex<T>, int> = 0>
real_type<T> trevc_protectsum(T a, T b, real_type<T> sf_max)
{
    real_type<T> xir = trevc_protectsum(real(a), real(b), sf_max);
    real_type<T> xii = trevc_protectsum(imag(a), imag(b), sf_max);
    return std::min<real_type<T>>(xir, xii);
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC_PROTECT_HH
