/// @file lae2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAE2_HH
#define TLAPACK_LAE2_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Computes the eigenvalues of a real symmetric 2x2 matrix A
 *  [ a b ]
 *  [ b c ]
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
 *
 * \verbatim
 *  s1 is accurate to a few ulps barring over/underflow.
 *
 *  s2 may be inaccurate if there is massive cancellation in the
 *  determinant a*c-b*b; higher precision or correctly rounded or
 *  correctly truncated arithmetic would be needed to compute s2
 *  accurately in all cases.
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
void lae2(T a, T b, T c, T& s1, T& s2)
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

    if (sm < zero) {
        s1 = half * (sm - rt);
        // Order of execution important.
        // To get fully accurate smaller eigenvalue,
        // next line needs to be executed in higher precision.
        s2 = (acmx / s1) * acmn - (b / s1) * b;
    }
    else if (sm > zero) {
        s1 = half * (sm + rt);
        // Order of execution important.
        // To get fully accurate smaller eigenvalue,
        // next line needs to be executed in higher precision.
        s2 = (acmx / s1) * acmn - (b / s1) * b;
    }
    else {
        // Includes case s1 = s2 = 0
        s1 = half * rt;
        s2 = -half * rt;
    }
}
}  // namespace tlapack

#endif  // TLAPACK_LAHQR_EIG22_HH
