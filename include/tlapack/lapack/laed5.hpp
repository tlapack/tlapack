/// @file laed5.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAED5_HH
#define TLAPACK_LAED5_HH

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
template <class d_t, class z_t, class delta_t, class real_t, class idx_t>
void laed5(idx_t i, d_t& d, z_t& z, delta_t& delta, real_t rho, real_t& dlam)

{
    real_t w, b, c, tau, temp;
    real_t del = d[1] - d[0];
    if (i == 0) {
        w = 1.0 + 2.0 * rho * (z[1] * z[1] - z[0] * z[0]) / del;

        if (w > 0) {
            b = del + rho * (z[0] * z[0] + z[1] * z[1]);
            c = rho * z[0] * z[0] * del;

            // B > 0, always

            tau = 2.0 * c / (b + sqrt(abs(b * b - 4 * c)));
            dlam = d[0] + tau;
            delta[0] = -z[0] / tau;
            delta[1] = z[1] / (del - tau);
        }
        else {
            b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
            c = rho * z[1] * z[1] * del;

            if (b > 0) {
                tau = -2.0 * c / (b + sqrt(b * b + 4 * c));
            }
            else {
                tau = (b - sqrt(b * b + 4 * c)) / 2.0;
            }

            dlam = d[1] + tau;
            delta[0] = -z[0] / (del + tau);
            delta[1] = -z[1] / tau;
        }

        temp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
        delta[0] = delta[0] / temp;
        delta[1] = delta[1] / temp;
    }
    else {
        // Now I = 2
        b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
        c = rho * z[1] * z[1] * del;

        if (b > 0) {
            tau = (b + sqrt(b * b + 4 * c)) / 2.0;
        }
        else {
            tau = 2.0 * c / (-b + sqrt(b * b + 4 * c));
        }

        dlam = d[1] + tau;
        delta[0] = -z[0] / (del + tau);
        delta[1] = -z[1] / tau;
        temp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
        delta[0] = delta[0] / temp;
        delta[1] = delta[1] / temp;
    }

    return;
}
}  // namespace tlapack

#endif  // TLAPACK_LAED5_HH