/// @file trevc_backsolve.hpp
/// @author Thijs Steel, KU Leuven, Belgium
// based on A. Schwarz et al., "Scalable eigenvector computation for the
// non-symmetric eigenvalue problem"
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC_BACKSOLVE_HH
#define TLAPACK_TREVC_BACKSOLVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/lapack/ladiv.hpp"
#include "tlapack/lapack/trevc_protect.hpp"
namespace tlapack {

/**
 * Calculate the k-th right eigenvector of T using backsubstitution.
 *
 * This is done by solving the triangular system
 *  (T - w*I)x = 0, where w is the k-th eigenvalue of T.
 *
 * This can be split into the block matrix:
 *             (k-1)    1   (n-k)
 * (k-1)  [ T11 - w*I  T12  T13      ] [x1]   [0]
 * 1      [ 0          0    T23      ] [x2] = [0]
 * (n-k)  [ 0          0    T33 - w*I] [x3]   [0]
 *
 * Assuming that T33 - w*I is invertible (i.e., w is not a repeated eigenvalue),
 * x3 = 0. (and even if it is not invertible, we can just choose x3 = 0)
 *
 * The first block row then gives:
 * (T11 - w*I)x1 = -T12*x2
 *
 * If we choose x2 = 1, we can solve for x1 using backsubstitution.
 *
 * The only special thing to take care of is that we don't want to modify T,
 * so we need to incorporate the shift -w*I during the backsubstitution.
 *
 * We should also handle potential overflow/underflow during the solve.
 * But this is not yet implemented.
 *
 * @param[in] T Upper quasi-triangular matrix
 * @param[out] v Vector to store the right eigenvector
 * @param[in] k Index of the eigenvector to compute
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          enable_if_t<is_real<type_t<matrix_T_t>>, int> = 0>
void trevc_backsolve_single(const matrix_T_t& T,
                            vector_v_t& v,
                            const size_type<matrix_T_t> k)
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

    // Initialize v to [-T(0:k-1, k), 1, 0]
    for (idx_t i = 0; i < k; ++i) {
        v[i] = -T(i, k);
    }
    v[k] = TT(1);
    for (idx_t i = k + 1; i < n; ++i) {
        v[i] = TT(0);
    }
    real_t scale = real_t(1);

    TT w = T(k, k);  // eigenvalue

    // Backsubstitution to solve the system
    auto T11 = slice(T, range(0, k), range(0, k));
    auto v1 = slice(v, range(0, k));

    // The matrix is real, so we need to consider potential
    // 2x2 blocks for complex conjugate eigenvalue pairs

    idx_t ii = 0;
    while (ii < k) {
        idx_t i = k - 1 - ii;
        bool is_2x2_block = false;
        if (i > 0) {
            if (T11(i, i - 1) != TT(0)) {
                is_2x2_block = true;
            }
        }

        if (is_2x2_block) {
            // 2x2 block
            // Solve the 2x2 system:
            // [T11(i-1,i-1)-w  T11(i-1,i)    ] [v1[i-1]] = [rhs1]
            // [T11(i,  i-1)    T11(i,  i)-w  ] [v1[i]  ]   [rhs2]

            real_t scale1a = trevc_protectsum(T11(i - 1, i - 1), -w, sf_max);
            real_t scale1b = trevc_protectsum(T11(i, i), -w, sf_max);
            real_t scale1 = std::min<real_t>(scale1a, scale1b);

            TT a = (scale1 * T11(i - 1, i - 1)) - (scale1 * w);
            TT b = (scale1 * T11(i - 1, i));
            TT c = (scale1 * T11(i, i - 1));
            TT d = (scale1 * T11(i, i)) - (scale1 * w);

            v1[i - 1] = scale1 * v1[i - 1];
            v1[i] = scale1 * v1[i];

            real_t scale2;
            trevc_2x2solve(a, b, c, d, v1[i - 1], v1[i], scale2, sf_min,
                           sf_max);

            // TODO: apply scale1 and scale2

            for (idx_t j = 0; j + 1 < i; ++j) {
                v1[j] -= T11(j, i - 1) * v1[i - 1];
                v1[j] -= T11(j, i) * v1[i];
            }

            ii += 2;
        }
        else {
            // 1x1 block

            // In the paper, they scale here so that denom cannot overflow
            // but this only happens if T11(i,i) and w are both very large
            // not just if the matrix is ill-conditioned.
            // The equations to take the scaling into account also don't
            // seem fully correct.
            TT denom = T11(i, i) - w;
            // Scale factor so that we can safely calculate v1[i] / (T11(i, i) -
            // w)
            real_t scale1 = trevc_protectdiv(v1[i], denom, sf_min, sf_max);

            // Safely execute the division
            v1[i] = (scale1 * v1[i]) / denom;

            if (scale1 != real_t(1)) {
                scale *= scale1;
                // Apply scale1 to all of v1
                for (idx_t j = 0; j < i; ++j) {
                    v1[j] = scale1 * v1[j];
                }
                for (idx_t j = i + 1; j < k; ++j) {
                    v1[j] = scale1 * v1[j];
                }
            }

            // Now update v1[0:i-1] -= T11(0:i-1, i) * v1[i]
            if (i > 0) {
                real_t ivmax = iamax(slice(v1, range(0, i)));
                real_t vmax = abs1(v1[ivmax]);

                // TODO: it is probably better to precompute these
                // and pass them as arguments
                real_t itmax = iamax(slice(col(T11, i), range(0, i)));
                real_t tmax = abs1(T11(itmax, i));

                real_t xnorm = abs1(v1[i]);

                // Scale factor so that
                // (scale2*v1[0:i-1]) - T11(0:i-1, i) * (scale2 * v1[i]) does
                // not overflow
                real_t scale2 = trevc_protectupdate(vmax, tmax, xnorm, sf_max);
                if (scale2 != real_t(1)) {
                    for (idx_t j = 0; j < i; ++j) {
                        v1[j] = (scale2 * v1[j]) - T11(j, i) * (scale2 * v1[i]);
                    }
                    // Apply scale2 to all of v1
                    for (idx_t j = i; j < k; ++j) {
                        v1[j] = scale2 * v1[j];
                    }

                    scale *= scale2;
                }
                else {
                    for (idx_t j = 0; j < i; ++j) {
                        v1[j] = (v1[j]) - T11(j, i) * (v1[i]);
                    }
                }
            }

            ii += 1;
        }
    }
}

/**
 * Calculate the k-th right eigenvector of T using backsubstitution.
 *
 * This is done by solving the triangular system
 *  (T - w*I)x = 0, where w is the k-th eigenvalue of T.
 *
 * This can be split into the block matrix:
 *             (k-1)    1   (n-k)
 * (k-1)  [ T11 - w*I  T12  T13      ] [x1]   [0]
 * 1      [ 0          0    T23      ] [x2] = [0]
 * (n-k)  [ 0          0    T33 - w*I] [x3]   [0]
 *
 * Assuming that T33 - w*I is invertible (i.e., w is not a repeated eigenvalue),
 * x3 = 0. (and even if it is not invertible, we can just choose x3 = 0)
 *
 * The first block row then gives:
 * (T11 - w*I)x1 = -T12*x2
 *
 * If we choose x2 = 1, we can solve for x1 using backsubstitution.
 *
 * The only special thing to take care of is that we don't want to modify T,
 * so we need to incorporate the shift -w*I during the backsubstitution.
 *
 * We should also handle potential overflow/underflow during the solve.
 * But this is not yet implemented.
 *
 * @param[in] T Upper quasi-triangular matrix
 * @param[out] v Vector to store the right eigenvector
 * @param[in] k Index of the eigenvector to compute
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          enable_if_t<is_complex<type_t<matrix_T_t>>, int> = 0>
void trevc_backsolve_single(const matrix_T_t& T,
                            vector_v_t& v,
                            const size_type<matrix_T_t> k)
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

    // Initialize v to [-T(0:k-1, k), 1, 0]
    for (idx_t i = 0; i < k; ++i) {
        v[i] = -T(i, k);
    }
    v[k] = TT(1);
    for (idx_t i = k + 1; i < n; ++i) {
        v[i] = TT(0);
    }
    real_t scale = real_t(1);

    TT w = T(k, k);  // eigenvalue

    // Backsubstitution to solve the system
    auto T11 = slice(T, range(0, k), range(0, k));
    auto v1 = slice(v, range(0, k));

    for (idx_t ii = 0; ii < k; ++ii) {
        idx_t i = k - 1 - ii;
        // Scale factor so that we can safely calculate T11(i, i) - w
        real_t scale1 = trevc_protectsum(T11(i, i), -w, sf_max);
        TT denom = (scale1 * T11(i, i)) - (scale1 * w);
        // Scale factor so that we can safely calculate v1[i] / (T11(i, i) - w)
        real_t scale2 = trevc_protectdiv(v1[i], denom, sf_min, sf_max);

        // Safely execute the division
        v1[i] = ladiv(scale2 * v1[i], denom);

        real_t scale12 = scale1 * scale2;
        if (scale12 != real_t(1)) {
            scale *= scale12;
            // Apply scale1 and scale2 to all of v1
            for (idx_t j = 0; j < i; ++j) {
                v1[j] = scale12 * v1[j];
            }
            for (idx_t j = i + 1; j < k; ++j) {
                v1[j] = scale12 * v1[j];
            }
        }

        // Now update v1[0:i-1] -= T11(0:i-1, i) * v1[i]
        if (i > 0) {
            real_t ivmax = iamax(slice(v1, range(0, i)));
            real_t vmax = abs1(v1[ivmax]);

            // TODO: it is probably better to precompute these
            // and pass them as arguments
            real_t itmax = iamax(slice(col(T11, i), range(0, i)));
            real_t tmax = abs1(T11(itmax, i));

            real_t xnorm = abs1(v1[i]);

            // Scale factor so that
            // (scale3*v1[0:i-1]) - T11(0:i-1, i) * (scale3 * v1[i]) does not
            // overflow
            real_t scale3 = trevc_protectupdate(vmax, tmax, xnorm, sf_max);
            if (scale3 != real_t(1)) {
                for (idx_t j = 0; j < i; ++j) {
                    v1[j] = (scale3 * v1[j]) - T11(j, i) * (scale3 * v1[i]);
                }
                // Apply scale3 to all of v1
                for (idx_t j = i; j < k; ++j) {
                    v1[j] = scale3 * v1[j];
                }

                scale *= scale3;
            }
            else {
                for (idx_t j = 0; j < i; ++j) {
                    v1[j] = (v1[j]) - T11(j, i) * (v1[i]);
                }
            }
        }
    }

    v[k] = scale * v[k];
}

/**
 * Calculate the k-th right eigenvector pair of T using
 * backsubstitution.
 *
 * This is done by solving the triangular system
 *  (T - w*I)x = 0, where w is the k-th eigenvalue of T.
 *
 * This can be split into the block matrix:
 *             (k-1)    1      1     (n-k-1)
 * (k-1)  [ T11 - w*I  T12     T13    T14      ] [x1]   [0]
 * 1      [ 0          alpha   beta   T24      ] [x2] = [0]
 * 1      [ 0          gamma   alpha  T34      ] [x3] = [0]
 * (n-k-1)[ 0          0       0      T44 - w*I] [x4]   [0]
 *
 * where we have assumed that the k-th eigenvalue is part of a complex
 * conjugate pair in normal form (i.e., w = alpha + i*sqrt(|beta * gamma|)).
 *
 * Like in the single eigenvector case, we assume that x4 = 0.
 *
 * If we choose x3 = i or x2 = 1, we can solve for x1 using backsubstitution.
 *
 * The only special thing to take care of is that we don't want to modify T,
 * so we need to incorporate the shift -w*I during the backsubstitution.
 *
 * @param[in] T Upper quasi-triangular matrix
 * @param[out] v_r Vector to store the real part of the right eigenvector
 * @param[out] v_i Vector to store the imaginary part of the right eigenvector
 * @param[in] k Index of the eigenvector to compute
 *              It is assumed that k and k+1 form a complex conjugate pair
 *              so k needs to be the first index of the 2x2 block, not the
 *              second.
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          enable_if_t<is_real<type_t<matrix_T_t>>, int> = 0>
void trevc_backsolve_double(const matrix_T_t& T,
                            vector_v_t& v_r,
                            vector_v_t& v_i,
                            const size_type<matrix_T_t> k)
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(T);

    tlapack_check(ncols(T) == n);
    tlapack_check(size(v_r) == n);
    tlapack_check(size(v_i) == n);
    tlapack_check(k < n);

    TT alpha = T(k, k);
    TT beta = T(k, k + 1);
    TT gamma = T(k + 1, k);

    // real part of eigenvalue
    TT wr = alpha;
    // imaginary part of eigenvalue
    TT wi = sqrt(abs(beta)) * sqrt(abs(gamma));

    // Depending of whether beta or gamma is bigger, we set x2 = 1 or x3 = i
    TT x2, x3;
    if (abs(beta) >= abs(gamma)) {
        x2 = TT(1);
        x3 = wi / beta;
    }
    else {
        x2 = -wi / gamma;
        x3 = TT(1);
    }

    // Initialize v_real and v_imag to -T(0:k-1, k:k+1) * [x2; i * x3]
    for (idx_t i = 0; i < k; ++i) {
        v_r[i] = -T(i, k) * x2;
        v_i[i] = -T(i, k + 1) * x3;
    }
    v_r[k] = x2;
    v_i[k] = TT(0);
    v_r[k + 1] = TT(0);
    v_i[k + 1] = x3;
    for (idx_t i = k + 2; i < n; ++i) {
        v_r[i] = TT(0);
        v_i[i] = TT(0);
    }

    // Now do a complex backsustitution using the shift wr + i*wi
    // but without forming complex numbers explicitly
    // on top of that, we need to take care of potential 2x2 blocks in T11
    auto T11 = slice(T, range(0, k), range(0, k));
    auto v1_r = slice(v_r, range(0, k));
    auto v1_i = slice(v_i, range(0, k));

    idx_t ii = 0;
    while (ii < k) {
        idx_t i = k - 1 - ii;
        bool is_2x2_block = false;
        if (i > 0) {
            if (T11(i, i - 1) != TT(0)) {
                is_2x2_block = true;
            }
        }

        if (is_2x2_block) {
            // 2x2 block

            // Solve the complex 2x2 system:
            // [T11(i-1,i-1)- (wr + i*wi)   T11(i-1,i)            ]
            // [T11(i,  i-1)               T11(i,  i)- (wr + i*wi)]
            // *
            // x
            // =
            // [v1_r[i-1] + i*v1_i[i-1]]
            // [v1_r[i]   + i*v1_i[i]  ]
            // Using real arithmetic only with Cramer's rule

            TT a11r = T11(i - 1, i - 1) - wr;
            TT a11i = -wi;
            TT a12 = T11(i - 1, i);
            TT a21 = T11(i, i - 1);
            TT a22r = T11(i, i) - wr;
            TT a22i = -wi;

            TT b1r = v1_r[i - 1];
            TT b1i = v1_i[i - 1];
            TT b2r = v1_r[i];
            TT b2i = v1_i[i];

            TT detr = a11r * a22r - a11i * a22i - a12 * a21;
            TT deti = a11r * a22i + a11i * a22r;

            TT denom = detr * detr + deti * deti;

            TT c1r = a22r * b1r - a22i * b1i - a12 * b2r;
            TT c1i = a22r * b1i + a22i * b1r - a12 * b2i;
            TT x1r = (c1r * detr + c1i * deti) / denom;
            TT x1i = (c1i * detr - c1r * deti) / denom;

            TT c2r = (a11r * b2r - a11i * b2i) - (a21 * b1r);
            TT c2i = (a11r * b2i + a11i * b2r) - (a21 * b1i);
            TT x2r = (c2r * detr + c2i * deti) / denom;
            TT x2i = (c2i * detr - c2r * deti) / denom;

            v1_r[i - 1] = x1r;
            v1_i[i - 1] = x1i;
            v1_r[i] = x2r;
            v1_i[i] = x2i;

            // Update the right-hand side
            for (idx_t j = 0; j + 1 < i; ++j) {
                // Real part
                v1_r[j] -= T11(j, i - 1) * v1_r[i - 1];
                v1_r[j] -= T11(j, i) * v1_r[i];

                // Imaginary part
                v1_i[j] -= T11(j, i - 1) * v1_i[i - 1];
                v1_i[j] -= T11(j, i) * v1_i[i];
            }

            ii += 2;
        }
        else {
            // 1x1 block

            // Do the complex division:
            // (v1_r[i] + i*v1_i[i]) / (T11(i, i) - (wr + i*wi))
            // in real arithmetic only
            TT a = v1_r[i];
            TT b = v1_i[i];
            TT c = T11(i, i) - wr;
            TT d = -wi;
            TT denom = c * c + d * d;
            v1_r[i] = (a * c + b * d) / denom;
            v1_i[i] = (b * c - a * d) / denom;

            // Update the right-hand side
            for (idx_t j = 0; j < i; ++j) {
                v1_r[j] -= T11(j, i) * v1_r[i];
                v1_i[j] -= T11(j, i) * v1_i[i];
            }

            ii += 1;
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC3_BACKSOLVE_HH
