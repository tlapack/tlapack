/// @file bdsqr.hpp Computes singular values and, optionally, the singular vectors of a n-by-n bidiagonal matrix B.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/bdsqr.h
//
// Copyright (c) 2014-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __BDSQR_HH__
#define __BDSQR_HH__

#include "lapack/types.hpp"

namespace lapack {

/** @brief Computes singular values and, optionally, the singular vectors
 * from the singular value decomposition (SVD) of a real n-by-n bidiagonal
 * matrix B using the implicit shift QR algorithm.
 *
 * The SVD of B has the form
 * \[
 *    B = Q S P^H
 * \]
 * where S is the diagonal matrix of singular values, Q is an orthogonal
 * matrix of left singular vectors, and P is an orthogonal matrix of right
 * singular vectors. If left singular vectors are requested, $U Q$ is
 * actually returned instead of Q, for a given scalar input matrix U. If
 * right singular vectors are requested, $P^H VT$ is actually returned instead
 * of $P^H$, for a given input matrix VT. When U and VT are the orthogonal
 * matrices that reduce a general matrix A to bidiagonal form: $A = U B VT$,
 * as computed by GEBRD, then
 * \[
 *    A = (U Q) S (P^H VT)
 * \]
 * is the SVD of A. Optionnally, $Q^H C$ may also be computed for a given scalar
 * matrix C.
 *
 * See "Computing  Small Singular Values of Bidiagonal Matrices With
 *   Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
 *   LAPACK Working Note #3 (or SIAM J. Sci. Statist. Comput. vol. 11,
 *   no. 5, pp. 873-912, Sept 1990)
 * @see http://www.netlib.org/lapack/lawnspdf/lawn03.pdf
 * and
 * "Accurate singular values and differential qd algorithms," by
 *   B. Parlett and V. Fernando, Technical Report CPAM-554, Mathematics
 *   Department, University of California at Berkeley, July 1992
 * for a detailed description of the algorithm.
 * 
 * @param[in] uplo
 *     What part of the matrix B is referenced:
 *     - Uplo::Lower: only the lower triangular part of B is referenced.
 *     - Uplo::Upper: only the upper triangular part of B is referenced.
 * @param[in] n    The order of the matrix B. n >= 0
 * @param[in] ncVT The number of columns of the matrix VT. ncVT >= 0
 * @param[in] nrU  The number of rows of the matrix U. nrU >= 0
 * @param[in] ncC  The number of columns of the matrix C. ncC >= 0
 * @param[in,out] d The diagonal elements of the bidiagonal matrix B.
 *   Real array of dimension min(m,n).
 *   On successful exit, d contains the singular values of B in decreasing
 *   order.
 * @param[in,out] e The off-diagonal elements of the bidiagonal matrix B.
 *   Real array of dimension min(m,n)-1.
 *   On successful exit, e has been modified.
 * @param[in,out] VT pointer to a scalar matrix.
 *   On entry, an n-by-ncVT matrix, if ncVT > 0.
 *   On successful exit, VT is overwritten by $P^H VT$.
 *   Not referenced if ncVT = 0.
 * @param[in] ldVT The column length of the matrix VT.
 *   ldVT >= n if ncVT > 0.
 * @param[in,out] U A pointer to a scalar matrix.
 *   On entry, an nrU-by-n scalar matrix, if nrU > 0.
 *   On succesful exit, U is overwritten by $U Q$.
 *   Not referenced if nrU = 0.
 * @param[in] ldU The column length of the matrix U.
 *   ldU >= nrU if nrU > 0.
 * @param[in,out] C A pointer to a scalar matrix.
 *   On entry, an n-by-ncC scalar matrix, if ncC > 0.
 *   On successful exit, C is overwritten by $Q^H C$.
 *   Not referenced if ncC = 0.
 * @param[in] ldC The column length of the matrix C.
 *   ldC >= n if ncC > 0.
 * 
 * @return 0 if success.
 * @return -i if the ith argument is invalid.
 * @return 1
 *   if ncVT = nrU = ncC = 0, then 1 element of e has not converged to
 *   zero,
 *   otherwise, a split was marked by a positive value in e.
 * @return 3
 *   if ncVT = nrU = ncC = 0, then 3 elements of e have not converged to
 *   zero,
 *   otherwise, the termination criterion of outer while loop is not met.
 * @return i if ncVT = nrU = ncC = 0, i elements have not converged
 *   to zero.
 *   d and e contain the elements of a bidiagonal matrix which is
 *   orthogonally similar to the input matrix B.
 *
 * @ingroup svd
 */
template< typename TB, typename TC >
void bdsqr(
    blas::Uplo uplo,
    blas::idx_t n, blas::idx_t ncVT, blas::idx_t nrU, blas::idx_t ncC,
    TB *d, TB *e,
    TB *VT, blas::idx_t ldVT,
    TB *U, blas::idx_t ldU,
    TC *C, blas::idx_t ldC,
    blas::real_type<TB,TC> *work )
{
    typedef blas::real_type<TB,TC> real_t;
    typedef blas::scalar_type<TB,TC> scalar_t;

    using blas::abs;
    using blas::max;
    using blas::min;
    using blas::pow;
    using blas::sqrt;

    // Constants
    const real_t zero(0.0);
    const real_t one(1.0);
    const real_t negone(-1.0);
    const real_t hundredth(0.01);
    const real_t ten(10.0);
    const real_t hundred(100.0);
    const real_t meigth(-0.125);

    // check arguments
    ( uplo != Uplo::Upper &&
                     uplo != Uplo::Lower, -1 );
    lapack_error_if( n < 0, -2 );
    lapack_error_if( ncVT < 0, -3 );
    lapack_error_if( nrU < 0, -4 );
    lapack_error_if( ncC < 0, -5 );
    lapack_error_if( ldVT < n, -9 );
    lapack_error_if( ldU < nrU, -11 );
    lapack_error_if( ldC < n, -13 );

    // Quick return
    if (n == 0) return 0;

    // maxiter controls the maximum number of passes of the algorithm through
    // its inner loop. The algorithm stops, and so fails to converge, if the
    // number of passes through the loop exceeds maxiter*n^2.
    const idx_t maxiter = 6;

    if (n > 1)
    {
        if (ncVT == 0 && nrU == 0 && ncC == 0)
        {
            // no singular vectors desired, use qd algorithm
            idx_t info;
            info = lasq1(n, d, e);
            if (info != 2)
                return info;
            // dqds didn't finish, try to finish
        }

        idx_t idir = 0;
        const idx_t nm1 = n - 1;
        const idx_t nm12 = nm1 + nm1;
        const idx_t nm13 = nm12 + nm1;
        const X eps = blas::uroundoff<X>();
        const X unfl = blas::safe_min<X>();

        if( uplo == Uplo::Lower )
        {
            // rotate matrix to be upper bidiagonal by applying Givens rotations
            // on the left
            real_t cs;
            real_t sn;
            real_t r;

            for (idx_t i = 0; i < n - 1; ++i)
            {
                lartg(d[i], e[i], cs, sn, r);
                d[i] = r;
                e[i] = sn * d[i + 1];
                d[i + 1] = cs * d[i + 1];
                work[i] = cs;
                work[nm1 + i] = sn;
            }
            if (nrU > 0)
                // update left singular vectors
                lasr('R', 'V', 'F', nrU, n, &work[0], &work[n - 1], U, ldU);
            if (ncC > 0)
                // update right singular vectors
                lasr('L', 'V', 'F', n, ncC, &work[0], &work[n - 1], C, ldC);
        }

        // tolmul controls the convergence criterion of the QR loop.
        // abs(tolmul) should be between 1 and 1/eps, and preferably between
        // 10 (for fast convergence) and .1/eps (for there be some accuracy in
        // the results).
        const real_t tolmul = max(ten, min(hundred, pow(eps, meigth)));

        // compute singular values to relative accuracy tol
        // By setting tol to be negative, the singular values will be computed
        // to absolute accuracy abs(tol)*norm(input matrix).
        const real_t tol = tolmul * eps;

        // compute approximate maximum, minimum singular values
        real_t thresh;
        real_t sminl = zero;
        real_t smax = zero;
        for (idx_t i = 0; i < n; ++i)
            smax = max(smax, abs(d[i]));
        for (idx_t i = 0; i < n - 1; ++i)
            smax = max(smax, abs(e[i]));
        if (tol >= zero)
        {
            // relative accuracy desired
            real_t sminoa = abs(d[0]);
            if (sminoa != zero)
            {
                real_t mu = sminoa;
                for (idx_t i = 1; i < n; ++i)
                {
                    mu = abs(d[i]) * (mu / (mu + abs(e[i - 1])));
                    sminoa = min(sminoa, mu);
                    if (sminoa == zero)
                        break;
                }
            }
            sminoa = sminoa / sqrt(real_t(n));
            thresh = max(tol * sminoa, real_t(maxiter * n * n) * unfl);
        }
        else
        {
            // absolute accuracy desired
            thresh = max(abs(tol) * smax, real_t(maxiter * n * n) * unfl);
        }

        // prepare for main iteration loop for the singular values
        // maxit is the maximum number of passes through the inner loop
        // permitted before nonconvergence is signaled
        const idx_t maxit = maxiter * n * n;
        idx_t iter = 0;

        // m points to last element of the unconverged part of the matrix
        // m and ll are 1-based
        idx_t m = n;
        idx_t ll = 1;
        idx_t oldll = -1;
        idx_t oldm = -1;

        real_t abss;
        real_t abse;
        real_t sigmn;
        real_t sigmx;
        real_t sinr;
        real_t cosr;
        real_t sinl;
        real_t cosl;
        real_t sll;
        real_t r;
        real_t cs;
        real_t sn;
        real_t oldcs;
        real_t oldsn;
        real_t f;
        real_t g;
        real_t shift;

        while (m > 1)
        {
            if (iter > maxit)
            {
                // failed to converge
                idx_t info = 0;
                for (idx_t i = 0; i < n - 1; ++i)
                    if (e[i] != zero)
                        info++;

                delete[] work;
                return info;
            }

            // find diagonal block of matrix to work on
            if (tol < zero && abs(d[m - 1]) <= thresh)
                d[m - 1] = zero;
            smax = abs(d[m - 1]);
            for (idx_t i = 1; i < m; ++i)
            {
                ll = m - i;
                abss = abs(d[ll - 1]);
                abse = abs(e[ll - 1]);
                if (tol < zero && abss <= thresh)
                    d[ll - 1] = zero;
                if (abse <= thresh)
                {
                    e[ll - 1] = zero;
                    ll++;
                    break;
                }
                smax = max(max(smax, abss), abse);
            }
            if (ll == m)
            {
                // convergence of bottom singular value
                m--;
                continue;
            }

            // now, e(ll) through e(m-1) are nonzero, e(ll-1) is zero if ll > 1

            if (ll == m - 1)
            {
                // 2-by-2 block, handle separately
                lasv2(d[m - 2], e[m - 2], d[m - 1], sigmn, sigmx, sinr, cosr, sinl, cosl);
                d[m - 2] = sigmx;
                e[m - 2] = zero;
                d[m - 1] = sigmn;

                if (ncVT > 0)
                    // update left singular vectors
                    rot(ncVT, &VT[m - 2], ldVT, &VT[m - 1], ldVT, cosr, sinr);
                if (nrU > 0)
                    // update right singular vectors
                    rot(nrU, &U[(m - 2) * ldU], 1, &U[(m - 1) * ldU], 1, cosl, sinl);
                if (ncC > 0)
                    // update C matrix
                    rot(ncC, &C[m - 2], ldC, &C[m - 1], ldC, cosl, sinl);

                m -= 2;
                continue;
            }

            if (ll > oldm || m < oldll)
            {
                // new submatrix, choose shift direction
                if (abs(d[ll - 1]) >= abs(d[m - 1]))
                    // chase bulge from top to bottom
                    idir = 1;
                else
                    // chase bulge from bottom to top
                    idir = 2;
            }

            if (idir == 1)
            {
                // run convergence test in forward direction
                // apply standard test to bottom of matrix
                if (abs(e[m - 2]) <= abs(tol * d[m - 1]) || (tol < zero && abs(e[m - 2]) <= thresh))
                {
                    e[m - 2] = zero;
                    continue;
                }
                if (tol >= zero)
                {
                    // relative accuracy desired
                    // apply convergence criterion forward
                    bool split = false;
                    real_t mu = abs(d[ll - 1]);
                    sminl = mu;
                    for (idx_t i = ll - 1; i < m - 1; ++i)
                    {
                        if (abs(e[i]) <= tol * mu)
                        {
                            e[i] = zero;
                            split = true;
                            break;
                        }
                        mu = abs(d[i + 1] * (mu / (mu + abs(e[i]))));
                        sminl = min(sminl, mu);
                    }
                    if (split)
                        continue;
                }
            }
            else
            {
                // run convergence test in backward direction
                // apply standard test to top of matrix
                if (abs(e[ll - 1]) <= abs(tol * d[ll - 1]) || (tol < zero && abs(e[ll - 1]) <= thresh))
                {
                    e[ll - 1] = zero;
                    continue;
                }
                if (tol >= zero)
                {
                    // relative accuracy desired
                    // apply convergence criterion backward
                    bool split = false;
                    real_t mu = abs(d[m - 1]);
                    sminl = mu;
                    for (idx_t i = m - 2; i > ll - 2; i--)
                    {
                        if (abs(e[i]) <= tol * mu)
                        {
                            e[i] = zero;
                            split = true;
                            break;
                        }
                        mu = abs(d[i] * (mu / (mu + abs(e[i]))));
                        sminl = min(sminl, mu);
                    }
                    if (split)
                        continue;
                }
            }

            oldll = ll;
            oldm = m;

            // Compute shift
            if (tol >= zero && n * tol * sminl <= max(eps, hundredth * tol) * smax)
            {
                // Use a zero shift to avoid loss of relative accuracy
                shift = zero;
            }
            else
            {
                // compute shift from 2-by-2 block at end of the matrix
                if (idir == 1)
                {
                    sll = abs(d[ll - 1]);
                    las2(d[m - 2], e[m - 2], d[m - 1], shift, r);
                }
                else
                {
                    sll = abs(d[m - 1]);
                    las2(d[ll - 1], e[ll - 1], d[ll], shift, r);
                }
                if (sll > zero)
                {
                    if (shift * shift < eps * sll * sll)
                    {
                        // the shift is negligible
                        shift = zero;
                    }
                }
            }

            iter = iter + m - ll;

            if (shift == zero)
            {
                // do simplified QR iteration

                if (idir == 1)
                {
                    // chase bulge from top to bottom
                    // save cosines and sines for later vector updates
                    cs = one;
                    oldcs = one;
                    for (idx_t i = ll - 1; i < m - 1; ++i)
                    {
                        lartg(d[i] * cs, e[i], cs, sn, r);
                        if (i > ll - 1)
                            e[i - 1] = oldsn * r;

                        lartg(oldcs * r, d[i + 1] * sn, oldcs, oldsn, d[i]);
                        work[i - ll + 1] = cs;
                        work[i - ll + 1 + nm1] = sn;
                        work[i - ll + 1 + nm12] = oldcs;
                        work[i - ll + 1 + nm13] = oldsn;
                    }
                    real_t h;
                    h = d[m - 1] * cs;
                    d[m - 1] = h * oldcs;
                    e[m - 2] = h * oldsn;

                    // update singular vectors
                    if (ncVT > 0)
                        lasr('L', 'V', 'F', m - ll + 1, ncVT, &work[0], &work[nm1],
                                     &VT[ll - 1], ldVT);
                    if (nrU > 0)
                        lasr('R', 'V', 'F', nrU, m - ll + 1, &work[nm12], &work[nm13],
                                     &U[(ll - 1) * ldU], ldU);
                    if (ncC > 0)
                        lasr('L', 'V', 'F', m - ll + 1, ncC, &work[nm12], &work[nm13],
                                     &C[ll - 1], ldC);

                    if (abs(e[m - 2]) <= thresh)
                        // convergence
                        e[m - 2] = zero;
                }
                else
                {
                    // chase bulge from bottom to top
                    // save cosines and sines for later vector updates
                    cs = one;
                    oldcs = one;
                    for (idx_t i = m - 1; i > ll - 1; i--)
                    {
                        lartg(d[i] * cs, e[i - 1], cs, sn, r);
                        if (i < m - 1)
                            e[i] = oldsn * r;

                        lartg(oldcs * r, d[i - 1] * sn, oldcs, oldsn, d[i]);
                        work[i - ll] = cs;
                        work[i - ll + nm1] = -sn;
                        work[i - ll + nm12] = oldcs;
                        work[i - ll + nm13] = -oldsn;
                    }
                    real_t h;
                    h = d[ll - 1] * cs;
                    d[ll - 1] = h * oldcs;
                    e[ll - 1] = h * oldsn;

                    // update singular vectors
                    if (ncVT > 0)
                        lasr('L', 'V', 'B', m - ll + 1, ncVT, &work[nm12], &work[nm13],
                                     &VT[ll - 1], ldVT);
                    if (nrU > 0)
                        lasr('R', 'V', 'B', nrU, m - ll + 1, &work[0], &work[n - 1],
                                     &U[(ll - 1) * ldU], ldU);
                    if (ncC > 0)
                        lasr('L', 'V', 'B', m - ll + 1, ncC, &work[0], &work[n - 1],
                                     &C[ll - 1], ldC);

                    if (abs(e[ll - 1]) <= thresh)
                        // convergence
                        e[ll - 1] = zero;
                }
            }
            else
            {
                // nonzero shift

                if (idir == 1)
                {
                    // chase bulge from top to bottom
                    // save cosines and sines for later singular vector updates

                    if (d[ll - 1] < zero)
                        f = (abs(d[ll - 1]) - shift) * (negone + shift / d[ll - 1]);
                    else
                        f = (abs(d[ll - 1]) - shift) * (one + shift / d[ll - 1]);
                    g = e[ll - 1];
                    for (idx_t i = ll - 1; i < m - 1; ++i)
                    {
                        lartg(f, g, cosr, sinr, r);
                        if (i > ll - 1)
                            e[i - 1] = r;
                        f = cosr * d[i] + sinr * e[i];
                        e[i] = cosr * e[i] - sinr * d[i];
                        g = sinr * d[i + 1];
                        d[i + 1] = cosr * d[i + 1];

                        lartg(f, g, cosl, sinl, r);
                        d[i] = r;
                        f = cosl * e[i] + sinl * d[i + 1];
                        d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                        if (i < m - 2)
                        {
                            g = sinl * e[i + 1];
                            e[i + 1] = cosl * e[i + 1];
                        }
                        work[i - ll + 1] = cosr;
                        work[i - ll + 1 + nm1] = sinr;
                        work[i - ll + 1 + nm12] = cosl;
                        work[i - ll + 1 + nm13] = sinl;
                    }
                    e[m - 2] = f;

                    //update singular vectors
                    if (ncVT > 0)
                        lasr('L', 'V', 'F', m - ll + 1, ncVT, &work[0], &work[nm1],
                                     &VT[ll - 1], ldVT);
                    if (nrU > 0)
                        lasr('R', 'V', 'F', nrU, m - ll + 1, &work[nm12], &work[nm13],
                                     &U[(ll - 1) * ldU], ldU);
                    if (ncC > 0)
                        lasr('L', 'V', 'F', m - ll + 1, ncC, &work[nm12], &work[nm13],
                                     &C[ll - 1], ldC);

                    if (abs(e[m - 2]) <= thresh)
                        // convergence
                        e[m - 2] = zero;
                }
                else
                {
                    // chase bulge from bottom to top
                    // save cosines and sines for later singular vector updates

                    if (d[m - 1] < zero)
                        f = (abs(d[m - 1]) - shift) * (negone + shift / d[m - 1]);
                    else
                        f = (abs(d[m - 1]) - shift) * (one + shift / d[m - 1]);
                    g = e[m - 2];
                    for (idx_t i = m - 1; i > ll - 1; i--)
                    {
                        lartg(f, g, cosr, sinr, r);
                        if (i < m - 1)
                            e[i] = r;
                        f = cosr * d[i] + sinr * e[i - 1];
                        e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                        g = sinr * d[i - 1];
                        d[i - 1] = cosr * d[i - 1];

                        lartg(f, g, cosl, sinl, r);
                        d[i] = r;
                        f = cosl * e[i - 1] + sinl * d[i - 1];
                        d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                        if (i > ll)
                        {
                            g = sinl * e[i - 2];
                            e[i - 2] = cosl * e[i - 2];
                        }
                        work[i - ll] = cosr;
                        work[i - ll + nm1] = -sinr;
                        work[i - ll + nm12] = cosl;
                        work[i - ll + nm13] = -sinl;
                    }
                    e[ll - 1] = f;

                    // update singular vectors
                    if (ncVT > 0)
                        lasr('L', 'V', 'B', m - ll + 1, ncVT, &work[nm12], &work[nm13],
                                     &VT[ll - 1], ldVT);
                    if (nrU > 0)
                        lasr('R', 'V', 'B', nrU, m - ll + 1, &work[0], &work[n - 1],
                                     &U[(ll - 1) * ldU], ldU);
                    if (ncC > 0)
                        lasr('L', 'V', 'B', m - ll + 1, ncC, &work[0], &work[n - 1],
                                     &C[ll - 1], ldC);

                    if (abs(e[ll - 1]) <= thresh)
                        // convergence
                        e[ll - 1] = zero;
                }
            }
        } // end of while(m>1) loop
        delete[] work;
    }

    // all singular values converged

    // make singular values positive
    for (idx_t i = 0; i < n; ++i)
    {
        if (d[i] < zero)
        {
            d[i] = -d[i];
            if (ncVT > 0)
                // change sign of singular vector
                scal(ncVT, negone, &VT[i], ldVT);
        }
    }

    // sort the singular values into decreasing order
    real_t smin;
    for (idx_t i = 0; i < n - 1; ++i)
    {
        // scan for the smallest d[i]
        idx_t isub = 0;
        smin = d[0];
        for (idx_t j = 1; j < n - i; ++j)
        {
            if (d[j] <= smin)
            {
                isub = j;
                smin = d[j];
            }
        }
        if (isub != n - 1 - i)
        {
            // swap singular values and vectors
            d[isub] = d[n - 1 - i];
            d[n - 1 - i] = smin;
            if (ncVT > 0)
                swap(ncVT, &VT[isub], ldVT, &VT[n - 1 - i], ldVT);
            if (nrU > 0)
                swap(nrU, &U[isub * ldU], 1, &U[(n - 1 - i) * ldU], 1);
            if (ncC > 0)
                swap(ncC, &C[isub], ldC, &C[n - 1 - i], ldC);
        }
    }

    return 0;
}

}

#endif // __BDSQR_HH__