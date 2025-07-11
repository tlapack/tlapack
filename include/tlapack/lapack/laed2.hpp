/// @file laed2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// TODO: Need to finished laed2

#ifndef TLAPACK_LAED2_HH
#define TLAPACK_LAED2_HH

//
#include "tlapack/base/utils.hpp"

//
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/iamax.hpp>
#include <tlapack/blas/rot.hpp>
#include <tlapack/blas/scal.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/laed4.hpp>
#include <tlapack/lapack/lamrg.hpp>
#include <tlapack/lapack/lapy2.hpp>

namespace tlapack {

template <class idx_t,
          class matrixQ_t,
          class d_t,
          class indxq_t,
          class real_t,
          class z_t,
          class dlambda_t,
          class w_t,
          class matrixQ2_t,
          class indx_t,
          class indxc_t,
          class indxp_t,
          class coltyp_t>
int laed2(idx_t& k,
          idx_t n,
          idx_t n1,
          d_t& d,
          matrixQ_t& q,
          idx_t ldq,
          indxq_t& indxq,
          real_t& rho,
          z_t z,
          dlambda_t& dlambda,
          w_t& w,
          matrixQ2_t& q2,
          indx_t& indx,
          indxc_t& indxc,
          indxp_t& indxp,
          coltyp_t& coltyp)
{
    std::cout << std::setprecision(15);
    using range = pair<idx_t, idx_t>;
    // Functor
    Create<matrixQ_t> new_matrix;

    int info = 0;

    if (n < 0) {
        info = -2;
    }
    else if (ldq < max(idx_t(1), n)) {
        info = -6;
    }
    else if (min(idx_t(1), (n / 2)) > n1 || (n / 2) < n1) {
        info = -3;
    }
    if (info != 0) {
        // Call Xerbla
        return info;
    }

    // Quick Return if Possible
    if (n == 0) {
        return info;
    }

    idx_t n2 = n - n1;
    idx_t n1p1 = n1 + 1;

    if (rho < 0) {
        auto zView = slice(z, range(n1p1 - 1, n));
        scal(real_t(-1.0), zView);
    }

    // Normalize z so that norm(z) = 1. Since z is the concatenation of a two
    // normalized vectors, norm2(z) = sqrt(2).
    real_t t = real_t(1.0 / sqrt(2.0));
    scal(t, z);

    // RHO = ABS(norm(z)**2 * RHO)
    rho = abs(real_t(2.0) * rho);

    // Sort the eigenvalues into increasing order
    for (idx_t i = n1p1 - 1; i < n; i++) {
        indxq[i] = indxq[i] + n1;
        std::cout << "INDXQ[I] =" << indxq[i] << std::endl;
    }

    // re-integrate the dflated parts from the last pass
    for (idx_t i = 0; i < n; i++) {
        dlambda[i] = d[indxq[i]];
        std::cout << "DLAMBDA[I] = " << dlambda[i] << std::endl;
    }
    lamrg(n1, n2, dlambda, 1, 1, indxc);
    for (idx_t i = 0; i < n; i++) {
        indx[i] = indxq[indxc[i]];
        std::cout << "INDXQ[INDXC[I]] = " << indxq[indxc[i]] << std::endl;
    }

    // Calculate the allowable deflation tolerance
    idx_t imax = iamax(z);
    idx_t jmax = iamax(d);

    real_t eps = ulp<real_t>() / real_t(2.);

    real_t tol = real_t(8.0) * eps * max(abs(d[jmax]), abs(z[imax]));

    // If the rank-1 modifier is small enough, everything is already deflated,
    // we set k to zero, no more needs to be done except to reorganize Q so that
    // its columns correspond with the elements in D.

    if (rho * abs(z[imax]) <= tol) {
        k = 0;
        idx_t iq2 = 0;
        for (idx_t j = 0; j < n; j++) {
            idx_t i = indx[j];

            auto qView = slice(q, range(0, n), i);
            auto q2View = slice(q2, range(iq2, n));
            copy(qView, q2View);

            dlambda[j] = d[i];
            iq2 = iq2 + n;
        }

        auto q2View = new_matrix(q2, n, n);
        lacpy(GENERAL, q2View, q);
        copy(dlambda, d);
        return info;
    }

    // If there are multiple eigenvalues then the problem deflates.  Here the
    // number of equal eigenvalues are found.  As each equal eigenvalue is
    // found, an elementary reflector is computed to rotate the corresponding
    // eigensubspace so that the corresponding components of Z are zero in this
    // new basis.

    for (idx_t i = 0; i < n1; i++) {
        coltyp[i] = real_t(0);
    }

    for (idx_t i = n1p1 - 1; i < n; i++) {
        coltyp[i] = real_t(2);
    }

    k = 0;
    idx_t k2 = n;
    idx_t pj = 0;
    real_t nj;
    bool deflate = false;
    for (idx_t j = 0; j < n; j++) {
        nj = indx[j];
        if (rho * abs(z[nj]) <= tol) {
            // Deflate due to small z component

            k2 = k2 - 1;
            coltyp[nj] = real_t(3.0);
            indxp[k2] = nj;
            if (j == n - 1) {
                deflate = true;
            }
        }
        else {
            pj = nj;
        }

        std::cout << "NJ = " << nj << std::endl;
        std::cout << "J = " << j << std::endl;
        std::cout << "K = " << k << std::endl;
        std::cout << "K2 = " << k2 << std::endl;
        std::cout << "PJ = " << pj << std::endl;

        // // LINE 80
        // if (!deflate) {
        //     if (j < n) {
        //         nj = indx[j + 1];
        //     }
        //     else {
        //         nj = 0;
        //     }
        //     std::cout << "NJ = " << nj << std::endl;
        //     if (j > n - 1) {
        //         break;
        //     }
        //     if (rho * abs(z[nj]) <= tol) {
        //         // Deflate due to small z component

        //         k2 = k2 - 1;
        //         coltyp[nj] = real_t(3);
        //         indxp[k2] = nj;
        //     }
        //     else {
        //         // Check fi eign values are close enought to allow deflation
        //         real_t s = z[pj];
        //         real_t c = z[nj];

        //         // Find sqrt(a**2 + b**2) without overflow or destructive
        //         // underflow
        //         auto tau = lapy2(c, s);
        //         t = d[nj] - d[pj];
        //         c = c / tau;
        //         s = -s / tau;

        //         if (abs(t * c * s) <= tol) {
        //             // Deflation is possible

        //             z[nj] = tau;
        //             z[pj] = real_t(0.0);
        //             if (coltyp[nj] != coltyp[pj]) {
        //                 coltyp[nj] = real_t(0);
        //             }
        //             coltyp[pj] = real_t(0);

        //             auto qView1 = slice(q, range(0, n), pj);
        //             auto qView2 = slice(q, range(0, n), nj);
        //             rot(qView1, qView2, c, s);

        //             t = d[pj] * (c * c) + d[nj] * (s * s);
        //             d[nj] = d[pj] * (s * s) + d[nj] * (c * c);
        //             d[pj] = t;
        //             k2 = k2 - 1;
        //             real_t i = 0;
        //             while (true) {
        //                 if (k2 + i < n) {
        //                     if (d[pj] < d[indxp[k2 + i]]) {
        //                         indxp[k2 + i - 1] = indxp[k2 + i];
        //                         indxp[k2 + i] = pj;
        //                         i = i + 1;
        //                         continue;
        //                     }
        //                     else {
        //                         indxp[k2 + i - 1] = pj;
        //                     }
        //                 }
        //                 else {
        //                     indxp[k2 + i - 1] = pj;
        //                 }

        //                 pj = nj;
        //                 break;
        //             }
        //         }
        //         else {
        //             dlambda[k - 1] = d[pj];
        //             w[k] = z[pj];
        //             indxp[k - 1] = pj;
        //             pj = nj;
        //             k = k + 1;
        //         }
        //     }
        // }
    }

    /////////////////////////////////////////////////////////////BREAKS HERE

    /////////////////////////////////////////////////////////////BREAKS HERE

    // Recored the last eigenvalue
    k = k + 1;
    dlambda[k] = d[pj];
    w[k] = z[pj];
    indxp[k] = pj;

    // Count up the total number of the various types of columns, then form
    // a permutation which positions the four column types into four uniform
    // groups (although one or more of these groups may be empty).

    std::vector<real_t> ctot(4);

    for (idx_t j = 0; j < 4; j++) {
        ctot[j] = real_t(0);
    }

    for (idx_t j = 0; j < n; j++) {
        idx_t ct = coltyp[j];
        ctot[ct] = ctot[ct] + real_t(1);
    }

    std::vector<real_t> psm(4);
    // // PSM(*) = Position in SubMatrix (of types 1 through 4)

    psm[0] = real_t(0);
    psm[1] = ctot[0];
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    k = n - ctot[3] - 1;

    // Fill out the INDXC array so that the permutation which it induces will
    // place all type-1 columns first, all type-2 columns next then all
    // type-3's, and finally all type-4's.

    for (idx_t j = 0; j < n; j++) {
        idx_t js = indxp[j];
        idx_t ct = coltyp[js];
        indx[psm[ct]] = js;
        indxc[psm[ct]] = j;
        psm[ct] = psm[ct] + real_t(1);
    }

    // Sort the eigenvalues and corresponding eigenvectors into DLAMBDA and Q2
    // respectively.  The eigenvalues/vectors which were not deflated go
    // into the first K slots of DLAMBDA and Q2 respectively while those
    // which were deflated go into the last N - K slots.

    idx_t i = 0;
    idx_t iq1 = 0;
    idx_t iq2 = (ctot[0] + ctot[1]) * n1;

    for (idx_t j = 0; j < ctot[0]; j++) {
        real_t js = indx[i];

        auto qView = slice(q, range(0, n1), js);
        auto q2View = slice(q2, range{iq1, iq1 + n1});
        copy(qView, q2View);

        z[i] = d[js];
        i = i + 1;
        iq1 = iq1 + n1;
    }

    for (idx_t j = 0; j < ctot[1]; j++) {
        real_t js = indx[i];

        auto qView = slice(q, range(0, n1), js);
        auto q2View = slice(q2, range(iq1, iq1 + n1));
        copy(qView, q2View);

        qView = slice(q, range(n1, n1 + n2), js);
        q2View = slice(q2, range(iq2, iq2 + n2));
        copy(qView, q2View);

        i = i + 1;
        iq1 = iq1 + n1;
        iq2 = iq2 + n2;
    }

    for (idx_t j = 0; j < ctot[2]; j++) {
        real_t js = indx[i];

        auto qView = slice(q, range(n1, n1 + n2), js);
        auto q2View = slice(q2, range(iq2, iq2 + n2));
        copy(qView, q2View);

        z[i] = d[js];
        i = i + 1;
        iq2 = iq2 + n2;
    }

    iq1 = iq2;

    for (idx_t j = 0; j < ctot[3]; j++) {
        idx_t js = indx[i];
        auto qView = slice(q, range(0, n), js);
        iq2 = iq2 + n;
        z[i] = d[js];
        i = i + 1;
    }

    // The deflated eigenvalues and their corresponding vectors go back into
    // the last N - K slots of D and Q respectively.

    // if (k < n - 1) {
    //     auto q2Mat = new_matrix(q2, n, n);
    //     auto q2View = slice(q2Mat, range(iq1, n), ctot[3]);
    //     auto q1View = slice(q, range(0, n), k);
    //     lacpy(GENERAL, q2View, q1View);

    //     copy(slice(z, range{k + 1, n - k}), slice(d, range{k + 1, n - k}));
    // }

    // Copy CTOT into COLTYP for referencing in DLAED3.

    for (idx_t j = 0; j < 4; j++) {
        coltyp[j] = ctot[j];
    }

    return info;
}
}  // namespace tlapack

#endif  // TLAPACK_LAED2_HH