/// @file trevc_forwardsolve.hpp
/// @author Thijs Steel, KU Leuven, Belgium
// based on A. Schwarz et al., "Scalable eigenvector computation for the
// non-symmetric eigenvalue problem"
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC_FORWARDSOLVE_HH
#define TLAPACK_TREVC_FORWARDSOLVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/lapack/ladiv.hpp"
#include "tlapack/lapack/trevc_protect.hpp"

namespace tlapack {

/**
 * Calculate the k-th left eigenvector of T using forward substitution.
 *
 * This is done by solving the triangular system
 *  v**H * (T - w*I) = 0, where w is the k-th eigenvalue of T.
 *
 * This can be split into the block matrix:
 *                           (k-1)    1   (n-k)
 *  [v1]**H  [ T11 - w*I  T12  T13      ]    [0] (k-1)
 *  [v2]     [ 0          0    T23      ]  = [0] 1
 *  [v3]     [ 0          0    T33 - w*I]    [0] (n-k)
 *
 * We choose v1 = 0
 *
 * The last block column then gives:
 * v3**H * (T33 - w*I) = -v2**H * T23
 *
 * If we choose v2 = 1, we can solve for v3 using forward substitution.
 *
 * The only special thing to take care of is that we don't want to modify T,
 * so we need to incorporate the shift -w*I during the forward substitution.
 *
 * @param[in] T Upper quasi-triangular matrix
 * @param[out] v Vector to store the left eigenvector
 * @param[in] k Index of the eigenvector to compute
 * @param[in] colN Infinity norms of the columns of T (to help with scaling)
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          TLAPACK_VECTOR vector_colN_t,
          enable_if_t<is_real<type_t<vector_colN_t>>, int> = 0,
          enable_if_t<is_real<type_t<matrix_T_t>>, int> = 0>
void trevc_forwardsolve_single(const matrix_T_t& T,
                               vector_v_t& v,
                               const size_type<matrix_T_t> k,
                               const vector_colN_t& colN)
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using real_t = real_type<TT>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(T);

    tlapack_check(ncols(T) == n);
    tlapack_check(size(v) == n);
    tlapack_check(k < n);

    real_t scale = real_t(1);

    const real_t sf_max = safe_max<real_t>();
    const real_t sf_min = safe_min<real_t>();

    // Initialize v to [0, 1, -T( k, k+1:n-1 )]
    for (idx_t i = 0; i < k; ++i) {
        v[i] = TT(0);
    }
    v[k] = TT(1);
    for (idx_t i = k + 1; i < n; ++i) {
        v[i] = -T(k, i);
    }

    TT w = T(k, k);  // eigenvalue

    // Forward substitution to solve the system
    auto T33 = slice(T, range(k + 1, n), range(k + 1, n));
    auto v3 = slice(v, range(k + 1, n));

    // The matrix is real, so we need to consider potential
    // 2x2 blocks for complex conjugate eigenvalue pairs
    idx_t i = 0;
    while (i < size(v3)) {
        bool is_2x2_block = false;
        if (i + 1 < size(v3)) {
            if (T33(i + 1, i) != TT(0)) {
                is_2x2_block = true;
            }
        }

        if (is_2x2_block) {
            // 2x2 block

            // Step 1: update
            idx_t ivmax = iamax(slice(v3, range(0, size(v3))));
            real_t vmax = abs1(v3[ivmax]);

            // Note, here we use the inf norm of the column of T, but really,
            // it should be the 1-norm of the column.
            real_t tnorm1 = colN[k + i];
            real_t tnorm2 = colN[k + i + 1];
            real_t scale1a =
                trevc_protectupdate(abs(v3[i]), tnorm1, vmax, sf_max);
            real_t scale1b =
                trevc_protectupdate(abs(v3[i + 1]), tnorm2, vmax, sf_max);
            real_t scale1 = min(scale1a, scale1b);

            if (scale1 != real_t(1)) {
                // Apply scale1 to all of v3
                for (idx_t j = 0; j < size(v3); ++j) {
                    v3[j] = scale1 * v3[j];
                }
                scale *= scale1;
            }

            for (idx_t j = 0; j < i; ++j) {
                v3[i] -= T33(j, i) * v3[j];
                v3[i + 1] -= T33(j, i + 1) * v3[j];
            }

            // Solve the 2x2 (transposed) system:
            // [T33(i,i)-w   T33(i+1,i)    ] [v3[i]  ] = [rhs1]
            // [T33(i,i+1)   T33(i+1,i+1)-w] [v3[i+1]]   [rhs2]

            TT a = T33(i, i) - w;
            TT b = T33(i + 1, i);
            TT c = T33(i, i + 1);
            TT d = T33(i + 1, i + 1) - w;

            TT scale2;
            trevc_2x2solve(a, b, c, d, v3[i], v3[i + 1], scale2, sf_min,
                           sf_max);

            // Apply scale2 to all of v3
            if (scale2 != real_t(1)) {
                scale *= scale2;
                for (idx_t j = 0; j < i; ++j) {
                    v3[j] = scale2 * v3[j];
                }
                for (idx_t j = i + 2; j < size(v3); ++j) {
                    v3[j] = scale2 * v3[j];
                }
            }

            i += 2;
        }
        else {
            //
            // 1x1 block
            //

            // Step 1: update
            idx_t ivmax = iamax(slice(v3, range(0, size(v3))));
            real_t vmax = abs1(v3[ivmax]);

            // Note, here we use the inf norm of the column of T, but really,
            // it should be the 1-norm of the column.
            real_t tnorm = colN[k + i];
            real_t scale1 =
                trevc_protectupdate(abs(v3[i]), tnorm, vmax, sf_max);

            if (scale1 != real_t(1)) {
                // Apply scale1 to all of v3
                for (idx_t j = 0; j < size(v3); ++j) {
                    v3[j] = scale1 * v3[j];
                }
                scale *= scale1;
            }

            for (idx_t j = 0; j < i; ++j) {
                v3[i] -= T33(j, i) * v3[j];
            }

            // Step 2: division

            real_t denom = T33(i, i) - w;
            real_t scale2 = trevc_protectdiv(v3[i], denom, sf_min, sf_max);
            v3[i] = (scale2 * v3[i]) / denom;

            if (scale2 != real_t(1)) {
                // Apply scale2 to all of v3
                for (idx_t j = 0; j < i; ++j) {
                    v3[j] = scale2 * v3[j];
                }
                for (idx_t j = i + 1; j < size(v3); ++j) {
                    v3[j] = scale2 * v3[j];
                }
                scale *= scale2;
            }

            i += 1;
        }
    }

    v[k] = scale * v[k];
}

/**
 * Complex version of trevc_forwardsolve_single
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          TLAPACK_VECTOR vector_colN_t,
          enable_if_t<is_real<type_t<vector_colN_t>>, int> = 0,
          enable_if_t<is_complex<type_t<matrix_T_t>>, int> = 0>
void trevc_forwardsolve_single(const matrix_T_t& T,
                               vector_v_t& v,
                               const size_type<matrix_T_t> k,
                               const vector_colN_t& colN)
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using real_t = real_type<TT>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(T);

    tlapack_check(ncols(T) == n);
    tlapack_check(size(v) == n);
    tlapack_check(k < n);

    const real_t sf_max = safe_max<real_t>();
    const real_t sf_min = safe_min<real_t>();

    real_t scale = real_t(1);

    // Initialize v to [0, 1, -T( k, k+1:n-1 )]
    for (idx_t i = 0; i < k; ++i) {
        v[i] = TT(0);
    }
    v[k] = TT(1);
    for (idx_t i = k + 1; i < n; ++i) {
        v[i] = -conj(T(k, i));
    }

    TT w = T(k, k);  // eigenvalue

    // Forward substitution to solve the system
    auto T33 = slice(T, range(k + 1, n), range(k + 1, n));
    auto v3 = slice(v, range(k + 1, n));

    // The matrix is complex, so there are no two-by-two blocks to
    // consider
    for (idx_t i = 0; i < size(v3); ++i) {
        // Step 1: update
        idx_t ivmax = iamax(slice(v3, range(0, size(v3))));
        real_t vmax = abs1(v3[ivmax]);

        // Note, here we use the inf norm of the column of T, but really,
        // it should be the 1-norm of the column.
        real_t tnorm = colN[k + i];
        real_t scale1 = trevc_protectupdate(abs1(v3[i]), tnorm, vmax, sf_max);

        if (scale1 != real_t(1)) {
            // Apply scale1 to all of v3
            for (idx_t j = 0; j < size(v3); ++j) {
                v3[j] = scale1 * v3[j];
            }
            scale *= scale1;
        }

        for (idx_t j = 0; j < i; ++j) {
            v3[i] -= conj(T33(j, i)) * v3[j];
        }

        // v3[i] = v3[i] / conj(T33(i, i) - w);

        // Step 2: division

        TT denom = conj(T33(i, i) - w);
        real_t scale2 = trevc_protectdiv(v3[i], denom, sf_min, sf_max);
        v3[i] = ladiv(scale2 * v3[i], denom);

        if (scale2 != real_t(1)) {
            // Apply scale2 to all of v3
            for (idx_t j = 0; j < i; ++j) {
                v3[j] = scale2 * v3[j];
            }
            for (idx_t j = i + 1; j < size(v3); ++j) {
                v3[j] = scale2 * v3[j];
            }
            scale *= scale2;
        }
    }

    v[k] = scale * v[k];
}

/**
 * Calculate the k-th left eigenvector pair of T using
 * forward substitution.
 *
 * This is done by solving the triangular system
 *  y(T - w*I) = 0, where w is the k-th eigenvalue of T.
 *
 * This can be split into the block matrix:
 *             (k-1)    1      1     (n-k-1)
 * (k-1)  [ v1 ]**H [ T11 - w*I  T12     T13    T14      ]   [0]
 * 1      [ v2 ]    [ 0          alpha   beta   T24      ] = [0]
 * 1      [ v3 ]    [ 0          gamma   alpha  T34      ] = [0]
 * (n-k-1)[ v4 ]    [ 0          0       0      T44 - w*I]   [0]
 *
 * where we have assumed that the k-th eigenvalue is part of a complex
 * conjugate pair in normal form (i.e., w = alpha + i*sqrt(|beta * gamma|)).
 *
 * Like in the single eigenvector case, we assume that y1 = 0.
 *
 * If we choose y3 = i or y2 = 1, we can solve for y4 using forward
 * substitution.
 *
 * The only special thing to take care of is that we don't want to modify T,
 * so we need to incorporate the shift -w*I during the forward substitution.
 *
 * We should also handle potential overflow/underflow during the solve.
 * But this is not yet implemented.
 *
 * @param[in] T Upper quasi-triangular matrix
 * @param[out] v_r Vector to store the real part of the left eigenvector
 * @param[out] v_i Vector to store the imaginary part of the left eigenvector
 * @param[in] k Index of the eigenvector to compute
 *              It is assumed that k and k+1 form a complex conjugate pair
 *              so k needs to be the first index of the 2x2 block, not the
 *              second.
 * @param[in] colN Infinity norms of the columns of T (to help with scaling)
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          TLAPACK_VECTOR vector_colN_t,
          enable_if_t<is_real<type_t<matrix_T_t>>, int> = 0>
void trevc_forwardsolve_double(const matrix_T_t& T,
                               vector_v_t& v_r,
                               vector_v_t& v_i,
                               const size_type<matrix_T_t> k,
                               const vector_colN_t& colN)
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(T);

    tlapack_check(ncols(T) == n);
    tlapack_check(size(v_r) == n);
    tlapack_check(size(v_i) == n);
    tlapack_check(k < n);

    const TT sf_max = safe_max<TT>();
    const TT sf_min = safe_min<TT>();

    TT alpha = T(k, k);
    TT beta = T(k, k + 1);
    TT gamma = T(k + 1, k);

    // real part of eigenvalue
    TT wr = alpha;
    // imaginary part of eigenvalue
    TT wi = sqrt(abs(beta)) * sqrt(abs(gamma));

    // Depending of whether beta or gamma is bigger, we set y2 = 1 or y3 = i
    TT y2, y3;
    if (abs(gamma) >= abs(beta)) {
        y2 = TT(1);
        y3 = -wi / gamma;
    }
    else {
        y2 = -wi / beta;
        y3 = -TT(1);
    }

    TT scale = TT(1);

    // Initialize v_real and v_imag to -[y2; i * y3]**H * T(k:k+1, k+2:n-1)
    for (idx_t i = 0; i < k; ++i) {
        v_r[i] = TT(0);
        v_i[i] = TT(0);
    }
    v_r[k] = y2;
    v_i[k] = TT(0);
    v_r[k + 1] = TT(0);
    v_i[k + 1] = y3;
    for (idx_t i = k + 2; i < n; ++i) {
        v_r[i] = -T(k, i) * y2;
        v_i[i] = -T(k + 1, i) * y3;
    }

    // Now do a complex forward substitution using the shift wr + i*wi
    // but without forming complex numbers explicitly
    // on top of that, we need to take care of potential 2x2 blocks in T44
    auto T44 = slice(T, range(k + 2, n), range(k + 2, n));
    auto v4_r = slice(v_r, range(k + 2, n));
    auto v4_i = slice(v_i, range(k + 2, n));

    idx_t i = 0;
    while (i < size(v4_r)) {
        bool is_2x2_block = false;
        if (i + 1 < size(v4_r)) {
            if (T44(i + 1, i) != TT(0)) {
                is_2x2_block = true;
            }
        }

        if (is_2x2_block) {
            // 2x2 block

            // Step 1: update
            idx_t ivrmax = iamax(slice(v4_r, range(0, size(v4_r))));
            TT vmax_r = abs1(v4_r[ivrmax]);
            idx_t ivimax = iamax(slice(v4_i, range(0, size(v4_i))));
            TT vmax_i = abs1(v4_i[ivimax]);
            TT vmax = vmax_r + vmax_i;
            // Note, here we use the inf norm of the column of T, but really,
            // it should be the 1-norm of the column.
            TT tnorm1 = colN[k + i];
            TT tnorm2 = colN[k + i + 1];
            TT scale1a = trevc_protectupdate(abs(v4_r[i]) + abs(v4_i[i]),
                                             tnorm1, vmax, sf_max);
            TT scale1b = trevc_protectupdate(
                abs(v4_r[i + 1]) + abs(v4_i[i + 1]), tnorm2, vmax, sf_max);
            TT scale1 = min(scale1a, scale1b);

            if (scale1 != TT(1)) {
                // Apply scale1 to all of v4_r and v4_i
                for (idx_t j = 0; j < size(v4_r); ++j) {
                    v4_r[j] = scale1 * v4_r[j];
                    v4_i[j] = scale1 * v4_i[j];
                }
                scale *= scale1;
            }

            for (idx_t j = 0; j < i; ++j) {
                v4_r[i] -= T44(j, i) * v4_r[j];
                v4_i[i] -= T44(j, i) * v4_i[j];
                v4_r[i + 1] -= T44(j, i + 1) * v4_r[j];
                v4_i[i + 1] -= T44(j, i + 1) * v4_i[j];
            }

            // Solve the (transposed) complex 2x2 system:
            // y**H
            // *
            // [T44(i,i)- (wr + i*wi) T44(i,i+1)                 ]
            // [T44(i+1,  i)          T44(i+1,  i+1)- (wr + i*wi)]
            // =
            // [v4_r[i] + i*v4_i[i]       ]
            // [v4_r[i+1]   + i*v4_i[i+1] ]

            TT a11r = T44(i, i) - wr;
            TT a11i = wi;
            // a12 and a21 are switched to transpose the system
            TT a12 = T44(i + 1, i);
            TT a21 = T44(i, i + 1);
            TT a22r = T44(i + 1, i + 1) - wr;
            TT a22i = wi;

            TT scale2;
            trevc_2x2solve(a11r, a11i, a12, TT(0), a21, TT(0), a22r, a22i,
                           v4_r[i], v4_i[i], v4_r[i + 1], v4_i[i + 1], scale2,
                           sf_min, sf_max);

            if (scale2 != TT(1)) {
                // Apply scale2 to all of v4_r and v4_i
                scale *= scale2;
                for (idx_t j = 0; j < i; ++j) {
                    v4_r[j] = scale2 * v4_r[j];
                    v4_i[j] = scale2 * v4_i[j];
                }
                for (idx_t j = i + 2; j < size(v4_r); ++j) {
                    v4_r[j] = scale2 * v4_r[j];
                    v4_i[j] = scale2 * v4_i[j];
                }
            }

            i += 2;
        }
        else {
            // 1x1 block

            // Step 1: update
            idx_t ivrmax = iamax(slice(v4_r, range(0, size(v4_r))));
            TT vmax_r = abs1(v4_r[ivrmax]);
            idx_t ivimax = iamax(slice(v4_i, range(0, size(v4_i))));
            TT vmax_i = abs1(v4_i[ivimax]);

            TT tnorm = colN[k + i];
            TT scale1a =
                trevc_protectupdate(abs(v4_r[i]), tnorm, vmax_r, sf_max);
            TT scale1b =
                trevc_protectupdate(abs(v4_i[i]), tnorm, vmax_i, sf_max);
            TT scale1 = min(scale1a, scale1b);

            if (scale1 != TT(1)) {
                // Apply scale1 to all of v4_r and v4_i
                for (idx_t j = 0; j < size(v4_r); ++j) {
                    v4_r[j] = scale1 * v4_r[j];
                    v4_i[j] = scale1 * v4_i[j];
                }
                scale *= scale1;
            }

            for (idx_t j = 0; j < i; ++j) {
                v4_r[i] -= T44(j, i) * v4_r[j];
                v4_i[i] -= T44(j, i) * v4_i[j];
            }

            // Do the complex division:
            // (v4_r[i] + i*v4_i[i]) / (T44(i, i) - (wr + i*wi))
            TT scale2 = trevc_protectdiv(v4_r[i], v4_i[i], T44(i, i) - wr, wi,
                                         sf_min, sf_max);

            TT tr, ti;
            ladiv(scale2 * v4_r[i], scale2 * v4_i[i], T44(i, i) - wr, wi, tr,
                  ti);
            v4_r[i] = tr;
            v4_i[i] = ti;

            if (scale2 != TT(1)) {
                // Apply scale2 to all of v4_r and v4_i
                for (idx_t j = 0; j < i; ++j) {
                    v4_r[j] = scale2 * v4_r[j];
                    v4_i[j] = scale2 * v4_i[j];
                }
                for (idx_t j = i + 1; j < size(v4_r); ++j) {
                    v4_r[j] = scale2 * v4_r[j];
                    v4_i[j] = scale2 * v4_i[j];
                }
                scale *= scale2;
            }

            i += 1;
        }
    }

    v_r[k] = scale * v_r[k];
    v_i[k] = scale * v_i[k];
    v_r[k + 1] = scale * v_r[k + 1];
    v_i[k + 1] = scale * v_i[k + 1];
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC_FORWARDSOLVE_HH
