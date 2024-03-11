/// @file laev2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAEV2_HH
#define TLAPACK_LAEV2_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Computes the eigenvalues and eigenvector of a real symmetric 2x2 matrix A
 *  [ a b ]
 *  [ b c ]
 *  On exit, the decomposition satisfies:
 *  [ cs  sn ] [ a b ] [ cs -sn ] = [ s1  0  ]
 *  [ -sn cs ] [ b c ] [ sn  cs ]   [ 0   s2 ]
 *  where cs*cs + sn*sn = 1.
 *
 * @param[in] a
 *      Element (0,0) of A.
 * @param[in] b
 *      Element (0,1) and (1,0) of A.
 * @param[in] c
 *      Element (1,1) of A.
 * @param[out] s1
 *      The eigenvalue of A with the largest absolute value.
 * @param[out] s2
 *      The eigenvalue of A with the smallest absolute value.
 * @param[out] cs
 *      The cosine of the rotation matrix.
 * @param[out] sn
 *      The sine of the rotation matrix.
 *
 * \verbatim
 *  s1 is accurate to a few ulps barring over/underflow.
 *
 *  s2 may be inaccurate if there is massive cancellation in the
 *  determinant a*c-b*b; higher precision or correctly rounded or
 *  correctly truncated arithmetic would be needed to compute s2
 *  accurately in all cases.
 *
 *  cs and sn are accurate to a few ulps barring over/underflow.
 *
 *  Overflow is possible only if s1 is within a factor of 5 of overflow.
 *  Underflow is harmless if the input data is 0 or exceeds
 *     underflow_threshold / macheps.
 * \endverbatim
 *
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SCALAR T>
void laev2(T a, T b, T c, T& s1, T& s2, T& cs, T& sn)
{
    // Constants
    const T zero(0);
    const T one(1);
    const T two(2);
    const T half(0.5);

    // Compute the eigenvalues
    T sm = a + c;
    T df = a - c;
    T adf = abs(df);
    T tb = b + b;
    T ab = abs(tb);
    T acmx, acmn;
    if (abs(a) > abs(c)) {
        acmx = a;
        acmn = c;
    }
    else {
        acmx = c;
        acmn = a;
    }

    T rt;
    if (adf > ab) {
        rt = adf * sqrt(one + square(ab / adf));
    }
    else if (adf < ab) {
        rt = ab * sqrt(one + square(adf / ab));
    }
    else {
        // This case includes case AB=ADF=0
        rt = ab * sqrt(two);
    }

    int sign1;
    if (sm < zero) {
        s1 = half * (sm - rt);
        sign1 = -1;
        // Order of execution important.
        // To get fully accurate smaller eigenvalue,
        // next line needs to be executed in higher precision.
        s2 = (acmx / s1) * acmn - (b / s1) * b;
    }
    else if (sm > zero) {
        s1 = half * (sm + rt);
        sign1 = 1;
        // Order of execution important.
        // To get fully accurate smaller eigenvalue,
        // next line needs to be executed in higher precision.
        s2 = (acmx / s1) * acmn - (b / s1) * b;
    }
    else {
        // Includes case s1 = s2 = 0
        s1 = half * rt;
        s2 = -half * rt;
        sign1 = 1;
    }

    // Compute the eigenvector
    int sign2;
    if (df >= zero) {
        cs = df + rt;
        sign2 = 1;
    }
    else {
        cs = df - rt;
        sign2 = -1;
    }
    T acs = abs(cs);
    if (acs > ab) {
        T ct = -tb / cs;
        sn = one / sqrt(one + square(ct));
        cs = ct * sn;
    }
    else {
        if (ab == zero) {
            cs = one;
            sn = zero;
        }
        else {
            T tn = -cs / tb;
            cs = one / sqrt(one + square(tn));
            sn = tn * cs;
        }
    }
    if (sign1 == sign2) {
        T tn = cs;
        cs = -sn;
        sn = tn;
    }
}
}  // namespace tlapack

#endif  // TLAPACK_LAEV2_HH
