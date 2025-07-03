/// @file laed6.hpp
/// @author Brian Dang, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAED6_HH
#define TLAPACK_LAED6_HH

//
#include "tlapack/base/utils.hpp"

namespace tlapack {

/** LAED6 used by STEDC. Computes one Newton step in solution of the secular
 * equation.
 *
 * LAED6 computes the positive or negative root (closest to the origin)
 * of
 *                  z(1)        z(2)        z(3)
 * f(x) =   rho + --------- + ---------- + ---------
 *                  d(1)-x      d(2)-x      d(3)-x
 *
 * It is assumed that
 *
 * if ORGATI = .true. the root is between d(2) and d(3);
 * otherwise it is between d(1) and d(2)
 *
 * This routine will be called by LAED4 when necessary. In most cases,
 * the root sought is the smallest in magnitude, though it might not be
 * in some extremely rare situations.
 *
 * @param[in] kniter
 *      KNITER is INTEGER
 *      Refer to LAED4 for its significance.
 * @param[in] orgati
 *      ORGATI is LOGICAL
 *      If ORGATI is true, the needed root is between d(2) and
 *      d(3); otherwise it is between d(1) and d(2).  See
 *      LAED4 for further details.
 * @param[in] rho
 *      RHO is DOUBLE PRECISION
 *      Refer to the equation f(x) above.
 * @param[out] d
 *      D is DOUBLE PRECISION array, dimension (3)
 *      D satisfies d(1) < d(2) < d(3).
 * @param[in] z
 *      Z is DOUBLE PRECISION array, dimension (3)
 *      Each of the elements in z must be positive.
 * @param[in] finit
 *       FINIT is DOUBLE PRECISION
 *       The value of f at 0. It is more accurate than the one
 *       evaluated inside this routine (if someone wants to do
 *       so).
 * @param[out] tau
 *       TAU is DOUBLE PRECISION
 *       The root of the equation f(x).
 * @return info
 *      INFO is INTEGER
 *       = 0:  successful exit
 *       > 0:  if INFO = 1, the updating process failed.
 *
 * @ingroup laed6
 */
template <class d_t, class z_t, class real_t, class idx_t>
int laed6(idx_t kniter,
          bool& orgati,
          real_t rho,
          d_t& d,
          z_t& z,
          real_t& finit,
          real_t& tau)

{
    idx_t niter;
    real_t lbd, ubd, temp, temp1, temp2, temp3, temp4, a, b, c, eta;
    real_t eps = ulp<real_t>();
    real_t maxit = 40;
    int info = 0;

    if (orgati) {
        lbd = d[1];
        ubd = d[2];
    }
    else {
        lbd = d[0];
        ubd = d[1];
    }

    if (finit < 0.0) {
        lbd = 0.0;
    }
    else {
        ubd = 0.0;
    }

    niter = 1;
    tau = 0;
    if (kniter == 1) {
        if (orgati) {
            temp = (d[2] - d[1]) / 2.0;
            c = rho + z[0] / ((d[0] - d[1]) - temp);
            a = c * (d[1] + d[2]) + z[1] + z[2];
            b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
        }
        else {
            temp = (d[0] - d[1]) / 2.0;
            c = rho + z[2] / ((d[2] - d[1]) - temp);
            a = c * (d[0] + d[1]) + z[0] + z[1];
            b = c * d[0] * d[2] + z[0] * d[1] + z[1] * d[0];
        }
        temp = max(max(abs(a), abs(b)), abs(c));
        a = a / temp;
        b = b / temp;
        c = c / temp;
        if (c == 0) {
            tau = b / a;
        }
        else if (a <= 0) {
            tau = (a - sqrt(abs(a * a - 4 * b * c))) / (2.0 * c);
        }
        else {
            tau = 2.0 * b / (a + sqrt(abs(a * a - 4 * b * c)));
        }

        if (tau < lbd || tau > ubd) {
            tau = (lbd + ubd) / 2.0;
        }

        if (d[0] == tau || d[1] == tau || d[2] == tau) {
            tau = 0.0;
        }
        else {
            temp = finit + tau * z[0] / (d[0] * (d[0] - tau)) +
                   tau * z[1] / (d[1] * (d[1] - tau)) +
                   tau * z[2] / (d[2] * (d[2] - tau));

            if (temp < 0.0) {
                lbd = tau;
            }
            else {
                ubd = tau;
            }
            if (abs(finit) <= abs(temp)) {
                tau = 0.0;
            }
        }
    }

    // get machine parameters for possible scaling to avoid overflow

    // modified by Sven: parameters SMALL1, SMINV1, SMALL2,
    // SMINV2, EPS are not SAVEd anymore between one call to the
    // others but recomputed at each call

    real_t base = 2.0;
    real_t safmin = std::numeric_limits<real_t>::min();
    real_t small1 = pow(base, log(safmin) / log(base) / 3.0);
    real_t sminv1 = 1.0 / small1;
    real_t small2 = small1 * small1;
    real_t sminv2 = sminv1 * sminv1;
    real_t sclfac, sclinv;
    std::vector<real_t> dscale(3);
    std::vector<real_t> zscale(3);

    // Determine if scaling of inputs necessary to avoid overflow when computing
    // 1/TEMP**3

    if (orgati) {
        temp = min(abs(d[1] - tau), abs(d[2] - tau));
    }
    else {
        temp = min(abs(d[0] - tau), abs(d[1] - tau));
    }

    bool scale = false;
    if (temp <= small1) {
        scale = true;
        if (temp <= small2) {
            // Scale up by power of radix nearest 1/SAFMIN**(2/3)
            sclfac = sminv2;
            sclinv = small2;
        }
        else {
            // Scale up by power of radix nearest 1/SAFMIN**(1/3)
            sclfac = sminv1;
            sclinv = small1;
        }

        for (idx_t i = 0; i < 3; i++) {
            dscale[i] = d[i] * sclfac;
            zscale[i] = z[i] * sclfac;
        }

        tau = tau * sclfac;
        lbd = lbd * sclfac;
        ubd = ubd * sclfac;
    }
    else {
        // Copy D and Z to DSCALE and ZSCALE
        for (idx_t i = 0; i < 3; i++) {
            dscale[i] = d[i];
            zscale[i] = z[i];
        }
    }

    real_t fc = 0;
    real_t df = 0;
    real_t ddf = 0;

    for (idx_t i = 0; i < 3; i++) {
        temp = 1.0 / (dscale[i] - tau);
        temp1 = zscale[i] * temp;
        temp2 = temp1 * temp;
        temp3 = temp2 * temp;
        fc = fc + temp1 / dscale[i];
        df = df + temp2;
        ddf = ddf + temp3;
    }

    real_t f = finit + tau * fc;
    bool converge = false;

    if (abs(f) <= 0.0) {
        converge = true;
    }
    if (f <= 0.0) {
        lbd = tau;
    }
    else {
        ubd = tau;
    }

    // Iteration begins -- Use Gragg-Thornton-Warner cubic convergent scheme

    // It is not hard to see that

    // 1) Iterations will go up monotonically
    // if FINIT < 0;

    // 2) Iterations will go down monotonically
    // if FINIT > 0.

    idx_t iter = niter + 1;

    while (iter < maxit) {
        if (converge) {
            break;
        }

        if (orgati) {
            temp1 = dscale[1] - tau;
            temp2 = dscale[2] - tau;
        }
        else {
            temp1 = dscale[0] - tau;
            temp2 = dscale[1] - tau;
        }

        a = (temp1 + temp2) * f - temp1 * temp2 * df;
        b = temp1 * temp2 * f;
        c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
        temp = max(max(abs(a), abs(b)), abs(c));
        a = a / temp;
        b = b / temp;
        c = b / temp;

        if (c == 0.0) {
            eta = b / a;
        }
        else if (a <= 0.0) {
            eta = (a - sqrt(abs(a * a - 4 * b * c))) / (2.0 * c);
        }
        else {
            eta = 2.0 * b / (a + sqrt(abs(a * a - 4 * b * c)));
        }

        if (f * eta >= 0.0) {
            eta = -f / df;
        }

        tau = tau + eta;
        if (tau < lbd || tau > ubd) {
            tau = (lbd + ubd) / 2.0;
        }

        fc = 0;
        real_t err = 0;
        df = 0;
        ddf = 0;

        for (idx_t i = 0; i < 3; i++) {
            if ((dscale[i] - tau) != 0) {
                temp = 1.0 / (dscale[i] - tau);
                temp1 = zscale[i] * temp;
                temp2 = temp1 * temp;
                temp3 = temp2 * temp;
                temp4 = temp1 / dscale[i];
                fc = fc + temp4;
                err = err + abs(temp4);
                df = df + temp2;
                ddf = ddf + temp3;
            }
            else {
                converge = true;
            }
        }

        if (!converge) {
            f = finit + tau * fc;
            err = 8.0 * (abs(finit) + abs(tau) * err) + abs(tau) * df;

            if ((abs(f) <= 4.0 * eps * err) ||
                ((ubd - lbd) <= 4.0 * eps * abs(tau))) {
                converge = true;
                break;
            }

            if (f <= 0.0) {
                lbd = tau;
            }
            else {
                ubd = tau;
            }
        }
        else {
            break;
        }
    }

    if (!converge) {
        info = 1;
    }

    // Undo scaling
    if (scale) {
        tau = tau * sclinv;
    }

    return info;
}
}  // namespace tlapack

#endif  // TLAPACK_LAED6_HH