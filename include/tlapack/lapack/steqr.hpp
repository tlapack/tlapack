/// @file steqr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zsteqr.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STEQR_HH
#define TLAPACK_STEQR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/lartg.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/lapack/lae2.hpp"
#include "tlapack/lapack/laev2.hpp"
#include "tlapack/lapack/lapy2.hpp"

namespace tlapack {

/**
 * STEQR computes all eigenvalues and, optionally, eigenvectors of a
 * hermitian tridiagonal matrix using the implicit QL or QR method.
 * The eigenvectors of a full or band hermitian matrix can also be found
 * if SYTRD or SPTRD or SBTRD has been used to reduce this matrix to
 * tridiagonal form.
 *
 * @return  0 if success
 *
 * @param[in] want_z bool
 *
 * @param[in,out] d Real vector of length n.
 *      On entry, diagonal elements of the bidiagonal matrix B.
 *      On exit, the singular values of B in decreasing order.
 *
 * @param[in,out] e Real vector of length n-1.
 *      On entry, off-diagonal elements of the bidiagonal matrix B.
 *      On exit, the singular values of B in decreasing order.
 *
 * @param[in,out] Z n-by-m matrix.
 *      On entry, the n-by-n unitary matrix used in the reduction
 *      to tridiagonal form.
 *      On exit, if info = 0, and want_z=true then Z contains the
 *      orthonormal eigenvectors of the original Hermitian matrix.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          class d_t,
          class e_t,
          enable_if_t<is_same_v<type_t<d_t>, real_type<type_t<d_t>>>, int> = 0,
          enable_if_t<is_same_v<type_t<e_t>, real_type<type_t<e_t>>>, int> = 0>
int steqr(bool want_z, d_t& d, e_t& e, matrix_t& Z)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // constants
    const real_t two(2);
    const real_t one(1);
    const real_t zero(0);
    const real_t three(3);
    const idx_t n = size(d);

    // Quick return if possible
    if (n == 0) return 0;
    if (n == 1) {
        if (want_z) Z(0, 0) = one;
        return 0;
    }

    // Determine the unit roundoff and over/underflow thresholds.
    const real_t eps = ulp<real_t>();
    const real_t eps2 = square(eps);
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = one / safmin;
    const real_t ssfmax = sqrt(safmax) / three;
    const real_t ssfmin = sqrt(safmin) / eps2;

    // Compute the eigenvalues and eigenvectors of the tridiagonal
    // matrix.
    const idx_t itmax = 30 * n;

    // istart and istop determine the active block
    idx_t istart = 0;
    idx_t istop = n;

    // Keep track of previous istart and istop to know when to change direction
    idx_t istart_old = -1;
    idx_t istop_old = -1;

    // If true, chase bulges from top to bottom
    // If false chase bulges from bottom to top
    // This variable is reevaluated for every new subblock
    bool forwarddirection = true;

    // Main loop
    for (idx_t iter = 0; iter < itmax; iter++) {
        if (iter == itmax) {
            // The QR algorithm failed to converge, return with error.
            return istop;
        }

        if (istop <= 1) {
            // All eigenvalues have been found, exit and return 0.
            break;
        }

        // Find active block
        for (idx_t i = istop - 1; i > istart; --i) {
            if (square(e[i - 1]) <=
                (eps2 * abs(d[i - 1])) * abs(d[i]) + safmin) {
                e[i - 1] = zero;
                istart = i;
                break;
            }
        }

        // An eigenvalue has split off, reduce istop and start the loop again
        if (istart == istop - 1) {
            istop = istop - 1;
            istart = 0;
            continue;
        }

        // A 2x2 block has split off, handle separately
        if (istart + 1 == istop - 1) {
            real_t s1, s2;
            if (want_z) {
                real_t cs, sn;
                laev2(d[istart], e[istart], d[istart + 1], s1, s2, cs, sn);

                auto z1 = col(Z, istart);
                auto z2 = col(Z, istart + 1);
                rot(z1, z2, cs, sn);
            }
            else {
                lae2(d[istart], e[istart], d[istart + 1], s1, s2);
            }
            d[istart] = s1;
            d[istart + 1] = s2;
            e[istart] = zero;

            istop = istop - 2;
            istart = 0;
            continue;
        }

        // Choose betwwen QL and QR iteration
        if (istart >= istop_old or istop <= istart_old) {
            forwarddirection = abs(d[istart]) > abs(d[istop - 1]);
        }
        istart_old = istart;
        istop_old = istop;

        // if (forwarddirection) {
        if (false) {
            // QR iteration

            // Form shift using last 2x2 block of the active matrix
            real_t p = d[istop - 1];
            real_t g = (d[istop - 2] - p) / (two * e[istop - 2]);
            real_t r = lapy2(g, one);
            g = d[istart] - p + e[istop - 2] / (g + (sgn(g) * r));

            real_t s = one;
            real_t c = one;
            p = zero;

            // Chase bulge from top to bottom
            for (idx_t i = istart; i < istop - 1; ++i) {
                real_t f = s * e[i];
                real_t b = c * e[i];
                lartg(g, f, c, s, r);
                if (i != istart) e[i - 1] = r;
                g = d[i] - p;
                r = (d[i + 1] - g) * s + two * c * b;
                p = s * r;
                d[i] = g + p;
                g = c * r - b;
                // If eigenvalues are desired, then apply rotations
                if (want_z) {
                    auto z1 = col(Z, i);
                    auto z2 = col(Z, i + 1);
                    rot(z1, z2, c, s);
                }
            }
            d[istop - 1] = d[istop - 1] - p;
            e[istop - 2] = g;
        }
        else {
            // QL iteration

            // Form shift using last 2x2 block of the active matrix
            real_t p = d[istart];
            real_t g = (d[istart + 1] - p) / (two * e[istart]);
            real_t r = lapy2(g, one);
            g = d[istop - 1] - p + e[istart] / (g + (sgn(g) * r));

            real_t s = one;
            real_t c = one;
            p = zero;

            // Chase bulge from bottom to top
            for (idx_t i = istop - 1; i > istart; --i) {
                real_t f = s * e[i - 1];
                real_t b = c * e[i - 1];
                lartg(g, f, c, s, r);
                if (i != istop - 1) e[i] = r;
                g = d[i] - p;
                r = (d[i - 1] - g) * s + two * c * b;
                p = s * r;
                d[i] = g + p;
                g = c * r - b;
                // If eigenvalues are desired, then apply rotations
                if (want_z) {
                    auto z1 = col(Z, i);
                    auto z2 = col(Z, i - 1);
                    rot(z1, z2, c, s);
                }
            }
            d[istart] = d[istart] - p;
            e[istart] = g;
        }
    }

    // Order eigenvalues and eigenvectors
    if (!want_z) {
        // Use quick sort
        // TODO: implement quick sort
    }
    else {
        // Use selection sort to minize swaps of eigenvectors
        for (idx_t i = 0; i < n - 1; ++i) {
            idx_t k = i;
            real_t p = d[i];
            for (idx_t j = i + 1; j < n; ++j) {
                if (d[j] < p) {
                    k = j;
                    p = d[j];
                }
            }
            if (k != i) {
                d[k] = d[i];
                d[i] = p;
                auto z1 = col(Z, i);
                auto z2 = col(Z, k);
                swap(z1, z2);
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_STEQR_HH
