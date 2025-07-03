/// @file laed4.hpp
/// @author Brian Dang, University of Colorado Denver, USA
//
// Copyright (c) 2025 University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAED4_HH
#define TLAPACK_LAED4_HH

//
#include "tlapack/base/utils.hpp"

//
#include <tlapack/lapack/laed5.hpp>
#include <tlapack/lapack/laed6.hpp>

namespace tlapack {

/** LAED4 used by STEDC. Finds a single root of the secular equation.
 *
 * This subroutine computes the I-th updated eigenvalue of a symmetric
 * rank-one modification to a diagonal matrix whose elements are
 * given in the array d, and that
 *
 *             D(i) < D(j)  for  i < j
 *
 * and that RHO > 0.  This is arranged by the calling routine, and is
 * no loss in generality.  The rank-one modified system is thus
 *
 *             diag( D )  +  RHO * Z * Z_transpose.
 *
 * where we assume the Euclidean norm of Z is 1.
 *
 * The method consists of approximating the rational functions in the
 * secular equation by simpler interpolating rational functions.
 *
 * @param[in] n
 *      N is INTEGER
 *      The length of all arrays.
 * @param[in] i
 *      I is INTEGER
 *      The index of the eigenvalue to be computed. 1 <= I <= N.
 * @param[in] d
 *      D is DOUBLE PRECISION array, dimension (N)
 *      The original eigenvalues.  It is assumed that they are in
 *      order, D(I) < D(J)  for I < J.
 * @param[in] z
 *      Z is DOUBLE PRECISION array, dimension (N)
 *      The components of the updating vector.
 * @param[out] delta
 *      DELTA is DOUBLE PRECISION array, dimension (N)
 *      If N > 2, DELTA contains (D(j) - lambda_I) in its  j-th
 *      component.  If N = 1, then DELTA(1) = 1. If N = 2, see LAED5
 *      for detail. The vector DELTA contains the information necessary
 *      to construct the eigenvectors by LAED3 and LAED9.
 * @param[in] rho
 *      RHO is DOUBLE PRECISION
 *      The scalar in the symmetric updating formula.
 * @param[out] dlam
 *      DLAM is DOUBLE PRECISION
 *      The computed lambda_I, the I-th updated eigenvalue.
 * @return info
 *      INFO is INTEGER
 *       = 0:  successful exit
 *       > 0:  if INFO = 1, the updating process failed.
 *
 *
 * Logical variable ORGATI (origin-at-i?) is used for distinguishing
 * whether D(i) or D(i+1) is treated as the origin.
 *
 *             ORGATI = .true.    origin at i
 *             ORGATI = .false.   origin at i+1
 *
 * Logical variable SWTCH3 (switch-for-3-poles?) is for noting
 * if we are working with THREE poles!
 *
 * MAXIT is the maximum number of iterations allowed for each
 * eigenvalue.
 *
 * @ingroup laed4
 */
template <class d_t, class z_t, class delta_t, class real_t, class idx_t>
int laed4(
    idx_t n, idx_t i, d_t& d, z_t& z, delta_t& delta, real_t rho, real_t& dlam)

{
    int info = 0;
    real_t psi, dpsi, phi, dphi, err, eta, a, b, c, w, del, tau, dltlb, dltub,
        temp;

    real_t maxIt = real_t(30);

    if (n == 1) {
        // Presumably, I = 1 upon entry
        dlam = real_t(d[0] + rho * z[0] * z[0]);
        delta[0] = real_t(1.0);
        return info;
    }
    else if (n == 2) {
        laed5(i, d, z, delta, rho, dlam);
        return info;
    }

    // Compute machine epsilon
    real_t eps = ulp<real_t>();
    // real_t eps = pow(2.0, -53);
    real_t rhoinv = 1.0 / rho;

    // The Case if i = n
    if (i == n - 1) {
        // Initialize some basic variables
        idx_t ii = n - 1;
        idx_t niter = 1;

        // Calculate Initial Guess
        real_t midpt = real_t(rho / 2.0);

        // If ||Z||_2 is not one, then TEMP should be set to RHO * ||Z||_2^2 /
        // TWO
        for (int j = 0; j < n; j++) {
            delta[j] = real_t((d[j] - d[i]) - midpt);
        }

        psi = 0;
        for (int j = 0; j < n - 2; j++) {
            psi += real_t(z[j] * z[j] / delta[j]);
        }

        c = rhoinv + psi;
        w = c + z[ii - 1] * z[ii - 1] / delta[ii - 1] +
            z[n - 1] * z[n - 1] / delta[n - 1];

        if (w <= 0) {
            real_t temp = z[n - 2] * z[n - 2] / (d[n - 1] - d[n - 2] + rho) +
                          z[n - 1] * z[n - 1] / rho;

            if (c <= temp) {
                tau = rho;
            }
            else {
                del = d[n - 1] - d[n - 2];
                a = -c * del + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
                b = z[n - 1] * z[n - 1] * del;

                if (a < 0) {
                    tau = 2 * b / (sqrt(a * a + 4.0 * b * c) - a);
                }
                else {
                    tau = (a + sqrt(a * a + 4.0 * b * c)) / (2 * c);
                }
            }

            // It can be proved that D(N)+RHO/2 <= LAMBDA(N) < D(N)+TAU <=
            // D(N)+RHO

            dltlb = midpt;
            dltub = rho;
        }
        else {
            del = d[n - 1] - d[n - 2];
            a = -c * del + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
            b = z[n - 1] * z[n - 1] * del;

            if (a < 0)
                tau = 2 * b / (sqrt(a * a + 4.0 * b * c) - a);
            else
                tau = (a + sqrt(a * a + 4.0 * b * c)) / (2.0 * c);

            // It can be proved that* D(N) < D(N) + TAU < LAMBDA(N) < D(N) + RHO
            // / 2 dltlb = 0.0;
            dltlb = 0;
            dltub = midpt;
        }

        for (idx_t j = 0; j < n; j++) {
            delta[j] = (d[j] - d[i]) - tau;
        }

        // Evaluate PSI and the derivative DPSI
        dpsi = 0, phi = 0, dphi = 0, err = 0;
        psi = 0;

        for (int j = 0; j < ii; j++) {
            real_t temp = z[j] / delta[j];
            psi += z[j] * temp;
            dpsi += temp * temp;
            err += psi;
        }

        err = abs(err);

        // Evaluate PHI and the derivative DPHI
        real_t temp = z[n - 1] / delta[n - 1];
        phi = z[n - 1] * temp;
        dphi = temp * temp;

        err = 8 * (-phi - psi) + err - phi + rhoinv + abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;

        // Test for convergence
        if (abs(w) <= eps * err) {
            dlam = d[i] + tau;
            return info;
        }

        if (w <= 0) {
            dltlb = max(dltlb, tau);
        }
        else {
            dltub = min(dltub, tau);
        }

        // Calculate the new step
        niter++;
        c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
        a = (delta[n - 2] + delta[n - 1]) * w -
            delta[n - 2] * delta[n - 1] * (dpsi + dphi);
        b = delta[n - 2] * delta[n - 1] * w;
        if (c < 0) {
            c = abs(c);
        }
        if (c == 0) {
            // ETA = B/A
            // ETA = RHO - TAU
            // ETA = DLTUB - TAU
            //
            // Update proposed by Li, Ren-Cang:
            eta = -w / (dpsi + dphi);
        }
        else if (a >= 0) {
            eta = (a + sqrt(abs(a * a - 4.0 * b * c))) / (2 * c);
        }
        else {
            eta = 2 * b / (a - sqrt(abs(a * a - 4.0 * b * c)));
        }

        // Note, eta should be positive if w is negative, and
        // eta should be negative otherwise. However,
        // if for some reason caused by roundoff, eta*w > 0,
        // we simply use one Newton step instead. This way
        // will guarantee eta*w < 0.

        if (w * eta > 0) {
            eta = -w / (dpsi + dphi);
        }

        temp = tau + eta;
        if (temp > dltub || temp < dltlb) {
            if (w < 0) {
                eta = (dltub - tau) / 2.0;
            }
            else {
                eta = (dltlb - tau) / 2.0;
            }
        }

        for (idx_t j = 0; j < n; j++) {
            delta[j] -= eta;
        }

        tau += eta;

        // Evaluate PSI and the derivative DPSI
        dpsi = 0;
        psi = 0;
        err = 0;
        for (idx_t j = 0; j < ii; j++) {
            temp = z[j] / delta[j];
            psi += z[j] * temp;
            dpsi += temp * temp;
            err += psi;
        }

        err = abs(err);

        // Evaluate PHI and the derivative DPHI
        temp = z[n - 1] / delta[n - 1];
        phi = z[n - 1] * temp;
        dphi = temp * temp;
        err =
            8.0 * (-phi - psi) + err - phi + rhoinv + abs(tau) * (dpsi + dphi);

        w = rhoinv + phi + psi;

        // Main loop to update the values of the array DELTA

        niter++;

        while (niter < maxIt) {
            // Test for convergence
            if (abs(w) <= eps * err) {
                dlam = d[i] + tau;
                return info;
            }

            if (w <= 0) {
                dltlb = max(dltlb, tau);
            }
            else {
                dltub = min(dltub, tau);
            }

            // Calculate the new step
            c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
            a = (delta[n - 2] + delta[n - 1]) * w -
                delta[n - 2] * delta[n - 1] * (dpsi + dphi);
            b = delta[n - 2] * delta[n - 1] * w;

            if (a >= 0) {
                eta = (a + sqrt(abs(a * a - 4.0 * b * c))) / (2.0 * c);
            }
            else {
                eta = 2.0 * b / (a - sqrt(abs(a * a - 4.0 * b * c)));
            }

            // Note, eta should be positive if w is negative, and eta should be
            // negative otherwise. However, if for some reason caused by
            // roundoff, eta*w > 0, we simply use one Newton step instead. This
            // way will guarantee eta*w < 0.

            if (w * eta > 0) {
                eta = -w / (dpsi + dphi);
            }
            temp = tau + eta;
            if (temp > dltub || temp < dltlb) {
                if (w < 0) {
                    eta = (dltub - tau) / 2.0;
                }
                else {
                    eta = (dltlb - tau) / 2.0;
                }
            }
            for (idx_t j = 0; j < n; j++) {
                delta[j] -= eta;
            }

            tau = tau + eta;

            // Evaluate PSI and the derivative DPSI
            dpsi = 0;
            psi = 0;
            err = 0;
            for (idx_t j = 0; j < ii; j++) {
                temp = z[j] / delta[j];
                psi += z[j] * temp;
                dpsi += temp * temp;
                err = err + psi;
            }
            err = abs(err);

            // Evaluate PHI and the derivative DPHI
            temp = z[n - 1] / delta[n - 1];
            phi = z[n - 1] * temp;
            dphi = temp * temp;
            err = 8.0 * (-phi - psi) + err - phi + rhoinv +
                  abs(tau) * (dpsi + dphi);
            w = rhoinv + phi + psi;

            niter++;
        }

        // Return with INFO = 1, NITER = MAXIT and not converged
        info = 1;
        dlam = d[i] + tau;
        return info;
    }
    else {
        // The case for 0 ≤ i < n
        idx_t niter = 1;
        idx_t ip1 = i + 1;

        // Calculate Inital Guess
        del = d[ip1] - d[i];
        real_t midpt = del / 2.0;
        for (idx_t j = 0; j < n; j++) {
            delta[j] = (d[j] - d[i]) - midpt;
        }

        psi = 0.0;
        for (idx_t j = 0; j < i; j++) {
            psi += z[j] * z[j] / delta[j];
        }

        phi = 0.0;
        for (idx_t j = n - 1; j >= i + 2; j--) {
            phi += z[j] * z[j] / delta[j];
        }

        c = rhoinv + psi + phi;

        w = c + z[i] * z[i] / delta[i] + z[ip1] * z[ip1] / delta[ip1];

        bool orgati;

        if (w > 0) {
            // d(i)< the ith eigenvalue < (d(i)+d(i+1))/2
            // We choose d(i) as origin.
            orgati = true;
            a = c * del + z[i] * z[i] + z[ip1] * z[ip1];

            b = z[i] * z[i] * del;

            if (a > 0) {
                tau = 2.0 * b / (a + sqrt(abs(a * a - 4.0 * b * c)));
            }
            else {
                tau = (a - sqrt(abs(a * a - 4.0 * b * c))) / (2.0 * c);
            }

            dltlb = 0.0;
            dltub = midpt;
        }
        else {
            // (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)
            // We choose d(i+1) as origin.
            orgati = false;
            a = c * del - z[i] * z[i] - z[ip1] * z[ip1];
            b = z[ip1] * z[ip1] * del;
            if (a < 0) {
                tau = 2.0 * b / (a - sqrt(abs(a * a + 4.0 * b * c)));
            }
            else {
                tau = -(a + sqrt(abs(a * a + 4 * b * c))) / (2.0 * c);
            }

            dltlb = -midpt;
            dltub = 0.0;
        }

        if (orgati) {
            for (idx_t j = 0; j < n; j++) {
                delta[j] = (d[j] - d[i]) - tau;
            }
        }
        else {
            for (idx_t j = 0; j < n; j++) {
                delta[j] = (d[j] - d[ip1]) - tau;
            }
        }

        idx_t ii;

        if (orgati) {
            ii = i;
        }
        else {
            ii = i + 1;
        }

        // if (iiml == std::numeric_limits<int>::max())
        idx_t iim1 = ii - 1;
        idx_t iip1 = ii + 1;

        // Evaluate PSI and the derivative DPSI
        psi = 0.0;
        dpsi = 0.0;
        err = 0.0;
        // for (idx_t j = 0; j <= iim1; j++) {
        for (idx_t j = 0; j + 1 <= ii; j++) {
            temp = z[j] / delta[j];
            psi = psi + z[j] * temp;
            dpsi += temp * temp;
            err += psi;
        }
        err = abs(err);

        // Evaluate PHI and the derivative DPHI
        phi = 0.0;
        dphi = 0.0;
        for (idx_t j = n - 1; j >= ip1; j--) {
            temp = z[j] / delta[j];
            phi += z[j] * temp;
            dphi += temp * temp;
            err += phi;
        }

        w = rhoinv + phi + psi;

        // W is the value of the secular function with
        // its ii-th element removed.

        bool swtch3 = false;
        if (orgati) {
            if (w < 0) {
                swtch3 = true;
            }
        }
        else {
            if (w > 0) {
                swtch3 = true;
            }
        }

        // if (ii == 1 || ii == n) {
        if (ii == 0 || ii == n - 1) {
            swtch3 = false;
        }

        temp = z[ii] / delta[ii];
        real_t dw = dpsi + dphi + temp * temp;
        temp = z[ii] * temp;
        w += temp;
        err = 8.0 * (phi - psi) + err + 2.0 * rhoinv + 3.0 * abs(temp) +
              abs(tau) * dw;

        // Test for Convergence
        if (abs(w) <= eps * err) {
            if (orgati) {
                dlam = d[i] + tau;
            }
            else {
                dlam = d[ip1] + tau;
            }
            return info;
        }

        if (w <= 0) {
            dltlb = max(dltlb, tau);
        }
        else {
            dltub = min(dltub, tau);
        }

        // Calculate the new step
        niter++;

        real_t zz[3];

        if (!swtch3) {
            if (orgati) {
                c = w - delta[ip1] * dw -
                    (d[i] - d[ip1]) * ((z[i] / delta[i]) * (z[i] / delta[i]));
            }
            else {
                c = w - delta[i] * dw -
                    (d[ip1] - d[i]) * ((z[ip1]) / delta[ip1]) *
                        ((z[ip1]) / delta[ip1]);
            }
            a = (delta[i] + delta[ip1]) * w - delta[i] * delta[ip1] * dw;
            b = delta[i] * delta[ip1] * w;
            if (c == 0) {
                if (a == 0) {
                    if (orgati) {
                        a = z[i] * z[i] +
                            delta[ip1] * delta[ip1] * (dpsi + dphi);
                    }
                    else {
                        a = z[ip1] * z[ip1] +
                            delta[i] * delta[i] * (dpsi + dphi);
                    }
                }
                eta = b / a;
            }
            else if (a <= 0) {
                eta = (a - sqrt(abs(a * a - 4.0 * b * c))) / (2.0 * c);
            }
            else {
                eta = 2.0 * b / (a + sqrt(abs(a * a - 4.0 * b * c)));
            }
        }
        else {
            // Interpolation using THREE most relevant poles
            temp = rhoinv + psi + phi;
            if (orgati) {
                real_t temp1 = z[iim1] / delta[iim1];
                temp1 = temp1 * temp1;
                c = temp - delta[iip1] * (dpsi + dphi) -
                    (d[iim1] - d[iip1]) * temp1;
                zz[0] = z[iim1] * z[iim1];
                zz[2] = delta[iip1] * delta[iip1] * ((dpsi - temp1) + dphi);
            }
            else {
                real_t temp1 = z[iip1] / delta[iip1];
                temp1 = temp1 * temp1;
                c = temp - delta[iim1] * (dpsi + dphi) -
                    (d[iip1] - d[iim1]) * temp1;
                zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - temp1));
                zz[2] = z[iip1] * z[iip1];
            }
            zz[1] = z[ii] * z[ii];
            info = laed6(niter, orgati, c, delta, zz, w, eta);

            if (info == 0) {
                return info;
            }
        }

        // Note, eta should be positive if w is negative, and
        // eta should be negative otherwise. However,
        // if for some reason caused by roundoff, eta*w > 0,
        // we simply use one Newton step instead. This way
        // will guarantee eta*w < 0.

        if (w * eta >= 0) {
            eta = -w / dw;
        }
        temp = tau + eta;
        if (temp > dltub || temp < dltlb) {
            if (w < 0) {
                eta = (dltub - tau) / 2;
            }
            else {
                eta = (dltlb - tau) / 2;
            }
        }

        real_t prew = w;

        for (idx_t j = 0; j < n; j++) {
            delta[j] -= eta;
        }

        // Evaluate PSI and the derivative DPSI
        psi = dpsi = err = 0;

        for (idx_t j = 0; j + 1 <= ii; j++) {
            temp = z[j] / delta[j];
            psi += z[j] * temp;
            dpsi += temp * temp;
            err += psi;
        }
        err = abs(err);

        // Evaluate PHI and the derivative DPHI
        phi = dphi = 0;
        for (idx_t j = n - 1; j >= iip1; j--) {
            temp = z[j] / delta[j];
            phi += z[j] * temp;
            dphi += temp * temp;
            err += phi;
        }

        temp = z[ii] / delta[ii];
        dw = dpsi + dphi + temp * temp;
        temp = z[ii] * temp;
        w = rhoinv + phi + psi + temp;
        err = 8 * (phi - psi) + err + 2.0 * rhoinv + 3.0 * abs(temp) +
              abs(tau + eta) * dw;

        real_t swtch = false;
        if (orgati) {
            if (-w > abs(prew) / 10.0) {
                swtch = true;
            }
        }
        else {
            if (w > abs(prew) / 10.0) {
                swtch = true;
            }
        }

        tau = tau + eta;

        // Main loop to update the values of the array DELTA

        real_t iter = niter + 1;

        while (iter < maxIt) {
            // Test for convergence
            if (abs(w) <= eps * err) {
                if (orgati) {
                    dlam = d[i] + tau;
                }
                else {
                    dlam = d[ip1] + tau;
                }
                return info;
            }

            if (w <= 0) {
                dltlb = max(dltlb, tau);
            }
            else {
                dltub = min(dltub, tau);
            }

            // Calculate the new step
            if (!swtch3) {
                if (!swtch) {
                    if (orgati) {
                        c = w - delta[ip1] * dw -
                            (d[i] - d[ip1]) * (z[i] / delta[i]) *
                                (z[i] / delta[i]);
                    }
                    else {
                        c = w - delta[i] * dw -
                            (d[ip1] - d[i]) * (z[ip1] / delta[ip1]) *
                                (z[ip1] / delta[ip1]);
                    }
                }
                else {
                    temp = z[ii] / delta[ii];
                    if (orgati) {
                        dpsi += temp * temp;
                    }
                    else {
                        dphi += temp * temp;
                    }
                    c = w - delta[i] * dpsi - delta[ip1] * dphi;
                }

                a = (delta[i] + delta[ip1]) * w - delta[i] * delta[ip1] * dw;
                b = delta[i] * delta[ip1] * w;

                if (c == 0) {
                    if (a == 0) {
                        if (!swtch) {
                            if (orgati) {
                                a = z[i] * z[i] +
                                    delta[ip1] * delta[ip1] * (dpsi + dphi);
                            }
                            else {
                                a = z[ip1] * z[ip1] +
                                    delta[i] * delta[i] * (dpsi + dphi);
                            }
                        }
                        else {
                            a = delta[i] * delta[i] * dpsi +
                                delta[ip1] * delta[ip1] * dphi;
                        }
                    }

                    eta = b / a;
                }
                else if (a < 0) {
                    eta = (a - sqrt(abs(a * a - 4.0 * b * c))) / (2.0 * c);
                }
                else {
                    eta = 2.0 * b / (a + sqrt(abs(a * a - 4.0 * b * c)));
                }
            }
            else {
                // Interpolation using THREE most relevant poles
                temp = rhoinv + psi + phi;
                if (swtch) {
                    c = temp - delta[iim1] * dpsi - delta[iip1] * dphi;
                    zz[0] = delta[iim1] * delta[iim1] * dpsi;
                    zz[2] = delta[iip1] * delta[iip1] * dphi;
                }
                else {
                    if (orgati) {
                        real_t temp1 = z[iim1] / delta[iim1];
                        temp1 = temp1 * temp1;
                        c = temp - delta[iip1] * (dpsi + dphi) -
                            (d[iim1] - d[iip1]) * temp1;
                        zz[0] = z[iim1] * z[iim1];
                        zz[2] =
                            delta[iip1] * delta[iip1] * ((dpsi - temp1) + dphi);
                    }
                    else {
                        real_t temp1 = z[iip1] / delta[iip1];
                        temp1 = temp1 * temp1;
                        c = temp - delta[iim1] * (dpsi + dphi) -
                            (d[iip1] - d[iim1]) * temp1;
                        zz[0] =
                            delta[iim1] * delta[iim1] * (dpsi + (dphi - temp1));
                        zz[2] = z[iip1] * z[iip1];
                    }
                }
                if (info == 0) {
                    return info;
                }
            }

            // Note, eta should be positive if w is negative, and
            // eta should be negative otherwise. However,
            // if for some reason caused by roundoff, eta*w > 0,
            // we simply use one Newton step instead. This way
            // will guarantee eta*w < 0.

            if (w * eta >= 0) {
                eta = -w / dw;
            }
            temp = tau + eta;
            if (temp > dltub || temp < dltlb) {
                if (w < 0) {
                    eta = (dltub - tau) / 2.0;
                }
                else {
                    eta = (dltlb - tau) / 2.0;
                }
            }

            for (idx_t j = 0; j < n; j++) {
                delta[j] = delta[j] - eta;
            }

            tau += eta;
            prew = w;

            // Evaluate PSI and the derivative DPSI
            psi = dpsi = err = 0;
            for (idx_t j = 0; j + 1 <= ii; j++) {
                temp = z[j] / delta[j];
                psi += z[j] * temp;
                dpsi += temp * temp;
                err += psi;
            }

            err = abs(err);

            // Evaluate PHI and the derivative DPHI
            phi = dphi = 0;
            for (idx_t j = n - 1; j >= iip1; j--) {
                temp = z[j] / delta[j];
                phi += z[j] * temp;
                dphi += temp * temp;
                err += phi;
            }

            temp = z[ii] / delta[ii];
            dw = dpsi + dphi + temp * temp;
            temp = z[ii] * temp;
            w = rhoinv + phi + psi + temp;
            err = 8 * (phi - psi) + err + 2.0 * rhoinv + 3.0 * abs(temp) +
                  abs(tau) * dw;

            if (w * prew > 0 && abs(w) > abs(prew) / 10.0) {
                swtch = !swtch;
            }

            iter++;
        }

        info = 1;
        if (orgati) {
            dlam = d[i] + tau;
        }
        else {
            dlam = d[ip1] + tau;
        }
    }

    return info;
}
}  // namespace tlapack

#endif  // TLAPACK_LAED4_HH