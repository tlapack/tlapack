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

#ifndef TLAPACK_STEQR3_HH
#define TLAPACK_STEQR3_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/lartg.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/lapack/lae2.hpp"
#include "tlapack/lapack/laev2.hpp"
#include "tlapack/lapack/lapy2.hpp"
#include "tlapack/lapack/rot_sequence3.hpp"

namespace tlapack {

/**
 * Options struct for steqr3
 */
struct Steqr3Opts {
    size_t nb = 32;  ///< Block size
};

/**
 * STEQR3 computes all eigenvalues and, optionally, eigenvectors of a
 * hermitian tridiagonal matrix using the implicit QL or QR method.
 * The eigenvectors of a full or band hermitian matrix can also be found
 * if SYTRD or SPTRD or SBTRD has been used to reduce this matrix to
 * tridiagonal form.
 *
 * STEQR3 is a variant of STEQR that uses rot_sequence3 to efficiently
 * accumulate the rotations into the eigenvector matrix.
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
int steqr3(
    bool want_z, d_t& d, e_t& e, matrix_t& Z, const Steqr3Opts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using real_matrix_t = real_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    // constants
    const real_t two(2);
    const real_t one(1);
    const real_t zero(0);
    const idx_t n = size(d);

    // Amount of rotation sequence to generate before applying it.
    const idx_t nb = opts.nb;

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

    // Compute the eigenvalues and eigenvectors of the tridiagonal
    // matrix.
    const idx_t itmax = 30 * n;

    // istart and istop determine the active block
    idx_t istart = 0;
    idx_t istop = n;

    // Keep track of previous istart and istop to know when to change direction
    idx_t istart_old = -1;
    idx_t istop_old = -1;

    // Keep track of the delayed rotation sequences
    idx_t i_block = 0;

    // Allocate matrices for storing rotations
    // TODO: use workspace instead of local allocation
    std::vector<real_t> C_((n - 1) * nb);
    std::vector<real_t> S_((n - 1) * nb);
    Create<real_matrix_t> new_real_matrix;
    auto C = new_real_matrix(C_, n - 1, nb);
    auto S = new_real_matrix(S_, n - 1, nb);
    // Initialize C and S to identity rotations
    for (idx_t j = 0; j < nb; ++j) {
        for (idx_t i = 0; i < n - 1; ++i) {
            C(i, j) = one;
            S(i, j) = zero;
        }
    }

    // If true, chase bulges from top to bottom
    // If false chase bulges from bottom to top
    // This variable is reevaluated for every new subblock
    bool forwarddirection = true;

    // Main loop
    for (idx_t iter = 0; iter < itmax; iter++) {
        if (want_z and (i_block >= nb or iter == itmax or istop <= 1)) {
            idx_t i_block2 = min<idx_t>(i_block + 1, nb);

            // Find smallest index where rotation is not identity
            idx_t i_start_block = n - 1;
            for (idx_t i = 0; i < i_block2; ++i) {
                for (idx_t j = 0; j < i_start_block; ++j)
                    if (C(0, i) != one or S(0, i) != zero) {
                        i_start_block = j;
                        break;
                    }
            }

            // Find largest index where rotation is not identity
            idx_t i_stop_block = 0;
            for (idx_t i = 0; i < i_block2; ++i) {
                for (idx_t j2 = n - 1; j2 > i_stop_block; --j2) {
                    idx_t j = j2 - 1;
                    if (C(j, i) != one or S(j, i) != zero) {
                        i_stop_block = j;
                        break;
                    }
                }
            }

            auto C2 = slice(C, range{i_start_block, i_stop_block + 1},
                            range{0, i_block2});
            auto S2 = slice(S, range{i_start_block, i_stop_block + 1},
                            range{0, i_block2});

            auto Z2 =
                slice(Z, range{0, n}, range{i_start_block, i_stop_block + 2});

            rot_sequence3(
                RIGHT_SIDE,
                forwarddirection ? Direction::Backward : Direction::Forward, C2,
                S2, Z2);
            // Reset block
            i_block = 0;

            // Initialize C and S to identity rotations
            for (idx_t j = 0; j < nb; ++j) {
                for (idx_t i = 0; i < n - 1; ++i) {
                    C(i, j) = one;
                    S(i, j) = zero;
                }
            }
        }
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

                // Store rotation
                C(istart, i_block) = cs;
                S(istart, i_block) = sn;
                // Normally, we would increment i_block here, but we do not
                // because the block has been deflated and should not interfere
                // with the next sequence.
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
            // forwarddirection = abs(d[istart]) > abs(d[istop - 1]);
            // For now, we only support forward direction
            forwarddirection = true;
        }
        istart_old = istart;
        istop_old = istop;

        if (forwarddirection) {
            // QR iteration

            // Form shift using last 2x2 block of the active matrix
            real_t p = d[istop - 1];
            real_t g = (d[istop - 2] - p) / (two * e[istop - 2]);
            real_t r = lapy2(g, one);
            g = d[istart] - p + e[istop - 2] / (real_t)(g + (sgn(g) * r));

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
                    // Store rotation for later
                    C(i, i_block) = c;
                    S(i, i_block) = s;
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
            g = d[istop - 1] - p + e[istart] / (real_t)(g + (sgn(g) * r));

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
                    // Store rotation for later
                    C(i - 1, i_block) = c;
                    S(i - 1, i_block) = s;
                }
            }
            d[istart] = d[istart] - p;
            e[istart] = g;
        }
        i_block++;
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
                tlapack::swap(z1, z2);
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_STEQR3_HH
