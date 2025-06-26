/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// This must run first
#include <tlapack/plugins/legacyArray.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include <tlapack/lapack/hetd2.hpp>
#include <tlapack/lapack/steqr.hpp>

using namespace tlapack;

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < n; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}
//------------------------------------------------------------------------------
// Computes the A = diag(d) + pzz^t
template <class d_t, class z_t, class delta_t, class real_t, class idx_t>
void laed4(
    idx_t n, idx_t i, d_t& d, z_t& z, delta_t& delta, real_t rho, real_t& dlam)

{
    real_t psi, dpsi, phi, dphi, err, eta, a, b, c, w, del, tau, dltlb, dltub;

    real_t maxIt = 30;

    real_t info = 0;

    if (n == 1) {
        // Presumably, I = 1 upon entry
        dlam = d[0] + rho * z[0] * z[0];
        delta[0] = 1;
        return;
    }
    else if (n == 2) {
        // Call DLAED5
        return;
    }

    // Compute machine epsilon
    real_t eps = std::numeric_limits<double>::epsilon();
    real_t rhoinv = 1 / rho;

    // The Case if i = n
    if (i == n) {
        /*
        // Initialize some basic variables
        idx_t ii = n - 1;
        idx_t niter = 1;

        // Calculate Initial Guess
        real_t midpt = rho / 2;

        // If ||Z||_2 is not one, then TEMP should be set to RHO * ||Z||_2^2 /
        // TWO
        for (int j = 0; j < n; j++) {
            delta[j] = (d[j] - d[i]) - midpt;
        }

        psi = 0;
        for (int j = 0; j < n - 2; j++) {
            psi += z[j] * z[j] / delta[j];
        }

        c = rhoinv + psi;
        w = c + z[ii] * z[ii] / delta[ii] + z[n - 1] * z[n - 1] / delta[n - 1];

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
                    tau = 2 * b / (sqrt(a * a + 4 * b * c) - a);
                }
                else {
                    tau = (a + sqrt(a * a + 4 * b * c)) / (2 * c);
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
                tau = 2 * b / (sqrt(a * a + 4 * b * c) - a);
            else
                tau = (a + sqrt(a * a + 4 * b * c)) / (2 * c);

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
            return;
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
            eta = (a + sqrt(abs(a * a - 4 * b * c))) / (2 * c);
        }
        else {
            eta = 2 * b / (a - sqrt(abs(a * a - 4 * b * c)));
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
                eta = (dltub - tau) / 2;
            }
            else {
                eta = (dltlb - tau) / 2;
            }
        }

        for (idx_t j = 0; j < n; j++) {
            delta[j] -= eta;
        }

        tau += eta;

        // Evaluate PSI and the derivative DPSI
        dpsi = psi = err = 0;
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
        err = 8 * (-phi - psi) + err - phi + rhoinv + abs(tau) * (dpsi + dphi);

        w = rhoinv + phi + psi;

        // Main loop to update the values of the array DELTA

        idx_t iter = niter + 1;

        while (niter < maxIt) {
            // Test for convergence
            if (abs(w) <= eps * err) {
                dlam += d[i] + tau;
                return;
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
                eta = (a + sqrt(abs(a * a - 4 * b * c))) / (2 * c);
            }
            else {
                eta = 2 * b / (a - sqrt(abs(a * a - 4 * b * c)));
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
                    eta = (dltub - tau) / 2;
                }
                else {
                    eta = (dltlb - tau) / 2;
                }
            }
            for (idx_t j = 0; j < n; j++) {
                delta[j] -= eta;
            }

            tau += eta;

            // Evaluate PSI and the derivative DPSI
            dpsi = psi = err = 0;
            for (idx_t j = 0; j < ii; j++) {
                temp = z[j] / delta[j];
                psi += z[j] / delta[j];
                dpsi += temp * temp;
                err += psi;
            }
            err = abs(err);

            // Evaluate PHI and the derivative DPHI
            temp = z[n - 1] / delta[n - 1];
            phi = z[n - 1] * temp;
            dphi = temp * temp;
            err = 8 * (-phi - psi) + err - phi + rhoinv +
                  abs(tau) * (dpsi + dphi);
            w = rhoinv + phi + psi;

            iter++;
        }

        // Return with INFO = 1, NITER = MAXIT and not converged
        info = 1;
        dlam = d[i] + tau;
        return;
        */
    }
    else {
        // The case for i < n
        idx_t niter = 1;
        idx_t ip1 = i + 1;

        // Calculate Inital Guess
        del = d[ip1] - d[i];
        real_t midpt = del / 2;
        for (idx_t j = 0; j < n; j++) {
            delta[j] = (d[j] - d[i]) - midpt;
        }

        psi = 0;
        for (idx_t j = 0; j < i; j++) {
            psi += z[j] * z[j] / delta[j];
        }

        phi = 0;
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
                tau = 2 * b / (a + sqrt(abs(a * a - 4 * b * c)));
            }
            else {
                tau = (a - sqrt(abs(a * a - 4 * b * c))) / (2 * c);
            }

            dltlb = 0;
            dltub = midpt;
        }
        else {
            // (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)
            // We choose d(i+1) as origin.
            orgati = false;
            a = c * del - z[i] * z[i] - z[ip1] * z[ip1];
            b = z[ip1] * z[ip1] * del;
            if (a < 0) {
                tau = 2 * b / (a - sqrt(abs(a * a + 4 * b * c)));
            }
            else {
                tau = -(a + sqrt(abs(a * a + 4 * b * c))) / (2 * c);
            }

            dltlb = -midpt;
            dltub = 0;
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

        idx_t iim1 = ii - 1;
        idx_t iip1 = ii + 1;

        // Evaluate PSI and the derivative DPSI
        psi = dpsi = err = 0;
        for (idx_t j = 0; j < iim1; j++) {
            real_t temp = z[j] / delta[j];
            psi += z[j] * temp;
            dpsi += temp * temp;
            err += psi;
        }
        err = abs(err);

        // Evaluate PHI and the derivative DPHI
        phi = dphi = 0;
        for (idx_t j = n - 1; j >= ip1; j--) {
            real_t temp = z[j] / delta[j];
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

        if (ii == 1 || ii == n) {
            swtch3 = false;
        }

        real_t temp = z[ii] / delta[ii];
        real_t dw = dpsi + dphi + temp * temp;
        temp = z[ii] * temp;
        w += temp;
        err =
            8 * (phi - psi) + err + 2 * rhoinv + 3 * abs(temp) + abs(tau) * dw;

        // Test for Convergence
        if (abs(w) <= eps * err) {
            if (orgati) {
                dlam = d[i] + tau;
            }
            else {
                dlam = d[ip1] + tau;
            }
            return;
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
                        ((z[ip1]) / delta[ip1]) * ((z[ip1]) / delta[ip1]);
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
                eta = (a - sqrt(abs(a * a - 4 * b * c))) / (2 * c);
            }
            else {
                eta = 2 * b / (a + sqrt(abs(a * a - 4 * b * c)));
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
                zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - temp1));
                zz[2] = z[iip1] * z[iip1];
            }
            zz[1] = z[ii] * z[ii];
            // call DLAED6
            if (info == 0) {
                return;
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

        for (idx_t j = 0; j < iim1; j++) {
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
        err = 8 * (phi - psi) + err + 2 * rhoinv + 3 * abs(temp) +
              abs(tau + eta) * dw;

        real_t swtch = false;
        if (orgati) {
            if (-w > abs(prew) / 10) {
                swtch = true;
            }
        }
        else {
            if (w > abs(prew) / 10) {
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
                return;
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
                    eta = (a - sqrt(abs(a * a - 4 * b * c))) / (2 * c);
                }
                else {
                    eta = 2 * b / (a + sqrt(abs(a * a - 4 * b * c)));
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
                // Call DLAED6
                if (info == 0) {
                    return;
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

            for (idx_t j = 0; j < n; j++) {
                delta[j] = delta[j] - eta;
            }

            tau += eta;
            prew = w;

            // Evaluate PSI and the derivative DPSI
            psi = dpsi = err = 0;
            for (idx_t j = 0; j < iim1; j++) {
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
            err = 8 * (phi - psi) + err + 2 * rhoinv + 3 * abs(temp) +
                  abs(tau) * dw;

            if (w * prew > 0 && abs(w) > abs(prew) / 10) {
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
}

//------------------------------------------------------------------------------
template <typename real_t>
void test_laed4(size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    using idx_t = size_type<matrix_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Vectors
    std::vector<real_t> d(n);
    std::vector<real_t> lamda(n);
    std::vector<real_t> e(n - 1);
    std::vector<real_t> u(n);
    std::vector<real_t> u_norm(n);
    std::vector<real_t> workSpace(n);
    std::vector<real_t> tau(n - 1);

    std::vector<real_t> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<real_t> Z_;
    auto Z = new_matrix(Z_, n, n);

    // Turn on for Debugging
    bool verbose = true;

    real_t rho = 2;

    srand(3);

    // Create Sorted d
    for (idx_t i = 0; i < n; i++) {
        d[i] = i + 1;
    }
    if (verbose) {
        std::cout << "\nSorted d = ( ";
        for (auto index : d) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // Create Random u
    for (idx_t i = 0; i < n; i++) {
        u[i] = rand();
    }
    if (verbose) {
        std::cout << "\nBefore u Norm = ( ";
        for (auto index : u) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }
    u_norm = u;

    // Normalize u
    real_t sum = 0;
    for (auto num : u_norm) {
        sum += num * num;
    }
    // u / sqrt(sum)
    for (idx_t i = 0; i < n; i++) {
        u_norm[i] = u_norm[i] / sqrt(sum);
    }
    if (verbose) {
        std::cout << "\nAfter u/sqrt(sum) = ( ";
        for (auto index : u_norm) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // uSum Check
    real_t uSum = 0;
    for (auto num : u_norm) {
        uSum += num * num;
    }
    if (verbose) {
        std::cout << "\nuSum should be 1: " << uSum << std::endl;
    }

    // Create u*u^T
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            A(i, j) = u_norm[i] * u_norm[j];
        }
    }
    if (verbose) {
        std::cout << "\nU*U^T Matrix =";
        printMatrix(A);
        std::cout << std::endl;
    }

    // Create A Matrix = rho * u*u^T
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            A(i, j) = rho * A(i, j);
        }
    }
    // Create A Matrix = D + rho * u*u^T
    for (idx_t i = 0; i < n; i++) {
        A(i, i) += d[i];
    }
    if (verbose) {
        std::cout << "\nA Matrix = D + rho * u*u^T";
        printMatrix(A);
        std::cout << std::endl;
    }

    // Turn A into a Tridiagonal
    hetd2(LOWER_TRIANGLE, A, tau);
    if (verbose) {
        std::cout << "\nA Matrix after hetd2 =";
        printMatrix(A);
        std::cout << std::endl;

        std::cout << "\ntau = ( ";
        for (auto index : tau) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // Get d from A
    for (idx_t i = 0; i < n; i++) {
        lamda[i] = A(i, i);
    }
    // Get e from A
    for (idx_t i = 0; i < n - 1; i++) {
        e[i] = A(i + 1, i);
    }
    if (verbose) {
        std::cout << "\nd = ( ";
        for (auto index : d) {
            std::cout << index << " ";
        }
        std::cout << ")\n";

        std::cout << "\ne = ( ";
        for (auto index : e) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    real_t dlam;
    laed4(n, static_cast<size_t>(1), d, u_norm, workSpace, rho, dlam);
    std::cout << "This is Lamda 1 :" << dlam << std::endl;
    laed4(n, static_cast<size_t>(2), d, u_norm, workSpace, rho, dlam);
    std::cout << "This is Lamda 2 :" << dlam << std::endl;
    laed4(n, static_cast<size_t>(3), d, u_norm, workSpace, rho, dlam);
    std::cout << "This is Lamda 3 :" << dlam << std::endl;

    // find the eigen and eigen vectors of the Tridiagonal A
    // steqr(false, d, e, A);
    steqr(true, lamda, e, Z);
    if (verbose) {
        std::cout << "\nlamda after steqr = ( ";
        for (auto index : lamda) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // f(lamda) = 1 + rho [(u_i^2 / (d_1 - lamda)) + (u_i^n / (d_n -
    // lamda))]
    for (idx_t i = 0; i < n; i++) {
        real_t f = 0;
        for (idx_t j = 0; j < n; j++) {
            f += (u_norm[j] * u_norm[j]) / (d[j] - lamda[i]);
        }
        f *= rho;
        f += 1;

        std::cout << "Lamda" << i << ":" << lamda[i] << " f:" << f << std::endl;
    }
}
//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n;

    // Default arguments
    n = (argc < 2) ? 7 : atoi(argv[1]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("test_laed4< float  >( %d, %d )", n, n);
    test_laed4<float>(n);
    printf("-----------------------\n");

    printf("test_laed4< double >( %d, %d )", n, n);
    test_laed4<double>(n);
    printf("-----------------------\n");

    printf("test_laed4< long double >( %d, %d )", n, n);
    test_laed4<long double>(n);
    printf("-----------------------\n");

    return 0;
}
