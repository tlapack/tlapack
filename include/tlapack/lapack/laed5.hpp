/// @file laed5.hpp
/// @author Brian Dang, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAED5_HH
#define TLAPACK_LAED5_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** LAED5 used by STEDC. Solves the 2-by-2 secular equation.
 *
 * This subroutine computes the I-th eigenvalue of a symmetric rank-one
 * modification of a 2-by-2 diagonal matrix
 *
 *             diag( D )  +  RHO * Z * transpose(Z) .
 *
 * The diagonal elements in the array D are assumed to satisfy
 *
 *             D(i) < D(j)  for  i < j .
 *
 * We also assume RHO > 0 and that the Euclidean norm of the vector
 * Z is one.
 *
 * @param[in] i
 *      I is INTEGER
 *      The index of the eigenvalue to be computed.  I = 1 or I = 2.
 * @param[in] d
 *      D is DOUBLE PRECISION array, dimension (2)
 *      The original eigenvalues.  We assume D(1) < D(2).
 * @param[in] z
 *      Z is DOUBLE PRECISION array, dimension (2)
 *      The components of the updating vector.
 * @param[out] delta
 *      DELTA is DOUBLE PRECISION array, dimension (2)
 *      The vector DELTA contains the information necessary
 *      to construct the eigenvectors.
 * @param[in] rho
 *      RHO is DOUBLE PRECISION
 *      The scalar in the symmetric updating formula.
 * @param[out] dlam
 *      DLAM is DOUBLE PRECISION
 *      The computed lambda_I, the I-th updated eigenvalue.
 *
 * @ingroup laed5
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