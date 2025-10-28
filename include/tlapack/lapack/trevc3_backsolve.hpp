/// @file trevc3_backsolve.hpp
/// @author Thijs Steel, KU Leuven, Belgium
// based on A. Schwarz et al., "Scalable eigenvector computation for the
// non-symmetric eigenvalue problem"
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC3_BACKSOLVE_HH
#define TLAPACK_TREVC3_BACKSOLVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"

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
 */
template <TLAPACK_MATRIX matrix_T_t, TLAPACK_VECTOR vector_v_t>
void trevc3_backsolve_single(const matrix_T_t& T,
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

    // Initialize v to [-T(0:k-1, k), 1, 0]
    for (idx_t i = 0; i < k; ++i) {
        v[i] = -T(i, k);
    }
    v[k] = TT(1);
    for (idx_t i = k + 1; i < n; ++i) {
        v[i] = TT(0);
    }

    TT w = T(k, k);  // eigenvalue

    // Backsubstitution to solve the system
    auto T11 = slice(T, range(0, k), range(0, k));
    auto v1 = slice(v, range(0, k));

    if constexpr (is_complex<TT>) {
        // The matrix is complex, so there are no two-by-two blocks to
        // consider

        for (idx_t ii = 0, i = k - 1; ii < k; ++ii, --i) {
            v1[i] = v1[i] / (T11(i, i) - w);

            for (idx_t j = 0; j < i; ++j) {
                v1[j] -= T11(j, i) * v1[i];
            }
        }
    }
    else {
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
                TT rhs1 = v1[i - 1];
                TT rhs2 = v1[i];

                TT a = T11(i - 1, i - 1) - w;
                TT b = T11(i - 1, i);
                TT c = T11(i, i - 1);
                TT d = T11(i, i) - w;

                TT det = a * d - b * c;

                v1[i - 1] = (d * rhs1 - b * rhs2) / det;
                v1[i] = (-c * rhs1 + a * rhs2) / det;

                for (idx_t j = 0; j + 1 < i; ++j) {
                    v1[j] -= T11(j, i - 1) * v1[i - 1];
                    v1[j] -= T11(j, i) * v1[i];
                }

                ii += 2;
            }
            else {
                // 1x1 block
                v1[i] = v1[i] / (T11(i, i) - w);

                for (idx_t j = 0; j < i; ++j) {
                    v1[j] -= T11(j, i) * v1[i];
                }

                ii += 1;
            }
        }
    }
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
 * We should also handle potential overflow/underflow during the solve.
 * But this is not yet implemented.
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          enable_if_t<is_real<type_t<matrix_T_t>>, int> = 0>
void trevc3_backsolve_double(const matrix_T_t& T,
                             vector_v_t& v_real,
                             vector_v_t& v_imag,
                             const size_type<matrix_T_t> k)
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(T);

    tlapack_check(ncols(T) == n);
    tlapack_check(size(v_real) == n);
    tlapack_check(size(v_imag) == n);
    tlapack_check(k < n);

    TT alpha = T(k, k);
    TT beta = T(k, k + 1);
    TT gamma = T(k + 1, k);

    // real part of eigenvalue
    TT wr = alpha;
    // imaginary part of eigenvalue
    TT wi = std::sqrt(std::abs(beta) * std::sqrt(std::abs(gamma)));

    // Depending of whether beta or gamma is bigger, we set x2 = 1 or x3 = i
    TT x2, x3;
    if (std::abs(beta) >= std::abs(gamma)) {
        x2 = TT(1);
        x3 = wi / beta;
    }
    else {
        x2 = -wi / gamma;
        x3 = TT(1);
    }

    // Initialize v_real and v_imag to -T(0:k-1, k:k+1) * [x2; i * x3]
    for (idx_t i = 0; i < k; ++i) {
        v_real[i] = -T(i, k) * x2;
        v_imag[i] = -T(i, k + 1) * x3;
    }

    // Now do a complex backsustitution using the shift wr + i*wi
    // but without forming complex numbers explicitly
    // on top of that, we need to take care of potential 2x2 blocks in T11
    auto T11 = slice(T, range(0, k), range(0, k));
    auto v1_real = slice(v_real, range(0, k));
    auto v1_imag = slice(v_imag, range(0, k));

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
        }
        else {
            // 1x1 block
            TT sum_real = TT(0);
            TT sum_imag = TT(0);
            for (idx_t j = i + 1; j < k; ++j) {
                sum_real += T11(i, j) * v1_real[j];
                sum_imag += T11(i, j) * v1_imag[j];
            }

            // Update v[i]
            TT denom = (T11(i, i) - wr);
            TT denom_sq = denom * denom + wi * wi;

            TT v_real_i = ((v1_real[i] - sum_real) * denom +
                           (v1_imag[i] - sum_imag) * wi) /
                          denom_sq;
            TT v_imag_i = ((v1_imag[i] - sum_imag) * denom -
                           (v1_real[i] - sum_real) * wi) /
                          denom_sq;

            v1_real[i] = v_real_i;
            v1_imag[i] = v_imag_i;

            ii += 1;
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC3_HH
