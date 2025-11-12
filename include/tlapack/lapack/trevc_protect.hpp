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
#include "tlapack/lapack/ladiv.hpp"

namespace tlapack {

/**
 * Given two numbers a and b, calculate a scaling factor alpha in (0, 1] such
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
 * Given two numbers a and b, calculate a scaling factor alpha in (0, 1] such
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
 * Given two numbers a and b, calculate a scaling factor alpha in (0, 1] such
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
 * @param ar Real part of numerator
 * @param ai Imaginary part of numerator
 * @param br Real part of denominator
 * @param bi Imaginary part of denominator
 * @param sf_min Safe minimum threshold (1/sf_max)
 * @param sf_max Safe maximum threshold
 *
 * @return Scaling factor alpha
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_real<T>, int> = 0>
real_type<T> trevc_protectdiv(
    T ar, T ai, T br, T bi, real_type<T> sf_min, real_type<T> sf_max)
{
    real_type<T> a1 = real_type<T>(2) * (abs(ar) + abs(ai));
    real_type<T> b1 = abs(br) + abs(bi);

    return trevc_protectdiv(a1, b1, sf_min, sf_max);
}

/**
 * Given the infinity norms of the matrices Y, T, and X, calculate a scaling
 * factor scale in (0, 1] such that the update Z = scale*Y - T * (scale * X)
 * does not overflow
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
 * @return Scaling factor
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
 * Calculate a scaling factor xi in [0.5, 1] such that the sum (xi * a) + (xi *
 * b) does not overflow
 *
 * @param a First addend
 * @param b Second addend
 * @param sf_max Safe maximum threshold
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
 * Calculate a scaling factor xi in [0.5, 1] such that the sum (xi * a) + (xi *
 * b) does not overflow
 *
 * @param a First addend
 * @param b Second addend
 * @param sf_max Safe maximum threshold
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

/**
 * Robustly solve a 2x2 system of equations
 * (a b) (x1) = scale * (rhs1)
 * (c d) (x2)           (rhs2)
 *
 * where scale is a scaling factor to avoid overflow during the solve
 *
 * @param a Coefficient a
 * @param b Coefficient b
 * @param c Coefficient c
 * @param d Coefficient d
 * @param x1 On input: right-hand side component rhs1
 *               On output: solution component x1
 * @param x2 On input: right-hand side component rhs2
 *               On output: solution component x2
 * @param scale On output: Scaling factor to avoid overflow
 * @param sf_min Safe minimum threshold
 * @param sf_max Safe maximum threshold
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_real<T>, int> = 0>
void trevc_2x2solve(
    T a, T b, T c, T d, T& x1, T& x2, T& scale, T sf_min, T sf_max)
{
    //
    // Step 1: compute LU factorization with complete pivoting
    //
    T abs_a = abs(a);
    T abs_b = abs(b);
    T abs_c = abs(c);
    T abs_d = abs(d);

    // 0 if a is largest
    // 1 if b is largest
    // 2 if c is largest
    // 3 if d is largest
    int ilargest = 0;
    T largest = abs_a;
    if (abs_b > largest) {
        ilargest = 1;
        largest = abs_b;
    }
    if (abs_c > largest) {
        ilargest = 2;
        largest = abs_c;
    }
    if (abs_d > largest) {
        ilargest = 3;
        largest = abs_d;
    }

    // Swap rows and columns to have the largest element in a
    bool col_swapped;
    if (ilargest == 0) {
        // a is largest, do nothing
        col_swapped = false;
    }
    else if (ilargest == 1) {
        // b is largest

        // swap columns
        std::swap(a, b);
        std::swap(c, d);
        col_swapped = true;
    }
    else if (ilargest == 2) {
        // c is largest

        // swap rows
        std::swap(a, c);
        std::swap(b, d);
        std::swap(x1, x2);
        col_swapped = false;
    }
    else {
        // d is largest

        // swap columns
        std::swap(a, b);
        std::swap(c, d);
        col_swapped = true;

        // swap rows
        std::swap(a, c);
        std::swap(b, d);
        std::swap(x1, x2);
    }

    // Step 2: LU factorization
    // (a b) -> (1    0) (u00 u01)
    // (c d)    (l10  1) (0   u11)
    // Note that we don't do overflow protection
    // for c/a or d - l10 * u01 here
    // I don't really see how that would work anyway

    T l10 = c / a;
    T u00 = a;
    T u01 = b;
    T u11 = d - l10 * u01;

    // Step 3: Solve Ly = scale1 * rhs
    T scale1 = trevc_protectupdate(abs(x2), abs(l10), abs(x1), sf_max);
    T y1 = scale1 * x1;
    T y2 = (scale1 * x2) - l10 * y1;

    // Step 4: Solve Ux = y
    T scale2 = trevc_protectdiv(y2, u11, sf_min, sf_max);
    y1 = scale2 * y1;
    y2 = scale2 * y2;
    x2 = y2 / u11;
    T scale3 = trevc_protectupdate(abs(y1), abs(u01), abs(x2), sf_max);
    x2 = scale3 * x2;
    y1 = scale3 * y1;
    y1 = y1 - u01 * x2;
    T scale4 = trevc_protectdiv(y1, u00, sf_min, sf_max);
    x1 = (scale4 * y1) / u00;

    // Step 5: undo column swap
    if (col_swapped) std::swap(x1, x2);

    scale = scale1 * scale2 * scale3 * scale4;
}

/**
 * Robustly solve a 2x2 system of equations
 * (ar + i*ai   br + i*bi) (x1r + i*x1i) = scale * (rhs1)
 * (cr + i*ci   dr + i*di) (x2r + i*x2i)           (rhs2)
 *
 * where scale is a scaling factor to avoid overflow during the solve
 *
 */
template <TLAPACK_SCALAR T, enable_if_t<is_real<T>, int> = 0>
void trevc_2x2solve(T ar,
                    T ai,
                    T br,
                    T bi,
                    T cr,
                    T ci,
                    T dr,
                    T di,
                    T& x1r,
                    T& x1i,
                    T& x2r,
                    T& x2i,
                    T& scale,
                    T sf_min,
                    T sf_max)
{
    //
    // Step 1: compute LU factorization with complete pivoting
    //
    T abs_a = abs(ar) + abs(ai);
    T abs_b = abs(br) + abs(bi);
    T abs_c = abs(cr) + abs(ci);
    T abs_d = abs(dr) + abs(di);

    // 0 if a is largest
    // 1 if b is largest
    // 2 if c is largest
    // 3 if d is largest
    int ilargest = 0;
    T largest = abs_a;
    if (abs_b > largest) {
        ilargest = 1;
        largest = abs_b;
    }
    if (abs_c > largest) {
        ilargest = 2;
        largest = abs_c;
    }
    if (abs_d > largest) {
        ilargest = 3;
        largest = abs_d;
    }

    // Swap rows and columns to have the largest element in a
    bool col_swapped;
    if (ilargest == 0) {
        // a is largest, do nothing
        col_swapped = false;
    }
    else if (ilargest == 1) {
        // b is largest

        // swap columns
        std::swap(ar, br);
        std::swap(ai, bi);
        std::swap(cr, dr);
        std::swap(ci, di);
        col_swapped = true;
    }
    else if (ilargest == 2) {
        // c is largest

        // swap rows
        std::swap(ar, cr);
        std::swap(ai, ci);
        std::swap(br, dr);
        std::swap(bi, di);
        std::swap(x1r, x2r);
        std::swap(x1i, x2i);
        col_swapped = false;
    }
    else {
        // d is largest

        // swap columns
        std::swap(ar, br);
        std::swap(ai, bi);
        std::swap(cr, dr);
        std::swap(ci, di);
        col_swapped = true;

        // swap rows
        std::swap(ar, cr);
        std::swap(ai, ci);
        std::swap(br, dr);
        std::swap(bi, di);
        std::swap(x1r, x2r);
        std::swap(x1i, x2i);
    }

    // Step 2: LU factorization
    // (a b) -> (1    0) (u00 u01)
    // (c d)    (l10  1) (0   u11)
    // Note that we don't do overflow protection
    // for c/a or d - l10 * u01 here
    // I don't really see how that would work anyway

    // l10 = c / a
    T l10r;
    T l10i;
    ladiv(cr, ci, ar, ai, l10r, l10i);
    // u00 = a
    T u00r = ar;
    T u00i = ai;
    // u01 = b
    T u01r = br;
    T u01i = bi;
    // u11 = d - l10 * u01
    T u11r = dr - (l10r * u01r - l10i * u01i);
    T u11i = di - (l10r * u01i + l10i * u01r);

    // Step 3: Solve Ly = scale1 * rhs
    // y1 = scale1 * rhs1
    // y2 = scale1 * rhs2 - l10 * (scale1 * rhs1)
    T scale1 = trevc_protectupdate(abs(x2r) + abs(x2i), abs(l10r) + abs(l10i),
                                   abs(x1r) + abs(x1i), sf_max);
    T y1r = scale1 * x1r;
    T y1i = scale1 * x1i;  // We could actually assume that x1i is zero, but i
                           // left it in to be general
    T y2r = (scale1 * x2r) - (l10r * y1r - l10i * y1i);
    T y2i = (scale1 * x2i) - (l10r * y1i + l10i * y1r);

    // Step 4: Solve Ux = y
    // x2 = y2 / u11
    T scale2 = trevc_protectdiv(y2r, y2i, u11r, u11i, sf_min, sf_max);
    y1r = scale2 * y1r;
    y1i = scale2 * y1i;
    y2r = scale2 * y2r;
    y2i = scale2 * y2i;
    ladiv(y2r, y2i, u11r, u11i, x2r, x2i);
    // x1 = (y1 - u01 * x2) / u00
    // temp = (y1 - u01 * x2)
    T scale3 = trevc_protectupdate(abs(y1r) + abs(y1i), abs(u01r) + abs(u01i),
                                   abs(x2r) + abs(x2i), sf_max);
    x2r = scale3 * x2r;
    x2i = scale3 * x2i;
    y1r = scale3 * y1r;
    y1i = scale3 * y1i;
    T tempr = y1r - (u01r * x2r - u01i * x2i);
    T tempi = y1i - (u01r * x2i + u01i * x2r);
    // x1 = temp / u00
    T scale4 = trevc_protectdiv(tempr, tempi, u00r, u00i, sf_min, sf_max);
    ladiv(scale4 * tempr, scale4 * tempi, u00r, u00i, x1r, x1i);

    // Step 5: undo column swap
    if (col_swapped) {
        std::swap(x1r, x2r);
        std::swap(x1i, x2i);
    }

    scale = scale1 * scale2 * scale3 * scale4;
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC_PROTECT_HH
