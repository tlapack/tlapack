/// @file trevc3_forwardsolve.hpp
/// @author Thijs Steel, KU Leuven, Belgium
// based on A. Schwarz et al., "Scalable eigenvector computation for the
// non-symmetric eigenvalue problem"
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC3_FORWARDSOLVE_HH
#define TLAPACK_TREVC3_FORWARDSOLVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"

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
 * We should also handle potential overflow/underflow during the solve.
 * But this is not yet implemented.
 */
template <TLAPACK_MATRIX matrix_T_t, TLAPACK_VECTOR vector_v_t>
void trevc3_forwardsolve_single(const matrix_T_t& T,
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

    if constexpr (is_complex<TT>) {
        // The matrix is complex, so there are no two-by-two blocks to
        // consider
        for (idx_t i = 0; i < size(v3); ++i) {
            for (idx_t j = 0; j < i; ++j) {
                v3[i] -= conj(T33(j, i)) * v3[j];
            }

            v3[i] = v3[i] / conj(T33(i, i) - w);
        }
    }
    else {
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

                for (idx_t j = 0; j < i; ++j) {
                    v3[i] -= T33(j, i) * v3[j];
                    v3[i + 1] -= T33(j, i + 1) * v3[j];
                }

                // Solve the 2x2 (transposed) system:
                // [T33(i,i)-w   T33(i+1,i)    ] [v3[i]  ] = [rhs1]
                // [T33(i,i+1)   T33(i+1,i+1)-w] [v3[i+1]]   [rhs2]
                TT rhs1 = v3[i];
                TT rhs2 = v3[i + 1];

                TT a = T33(i, i) - w;
                TT b = T33(i + 1, i);
                TT c = T33(i, i + 1);
                TT d = T33(i + 1, i + 1) - w;

                TT det = a * d - b * c;

                v3[i] = (d * rhs1 - b * rhs2) / det;
                v3[i + 1] = (-c * rhs1 + a * rhs2) / det;

                i += 2;
            }
            else {
                // 1x1 block
                for (idx_t j = 0; j < i; ++j) {
                    v3[i] -= T33(j, i) * v3[j];
                }

                v3[i] = v3[i] / (T33(i, i) - w);

                i += 1;
            }
        }
    }
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
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_VECTOR vector_v_t,
          enable_if_t<is_real<type_t<matrix_T_t>>, int> = 0>
void trevc3_forwardsolve_double(const matrix_T_t& T,
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
    TT wi = std::sqrt(std::abs(beta)) * std::sqrt(std::abs(gamma));

    // Depending of whether beta or gamma is bigger, we set y2 = 1 or y3 = i
    TT y2, y3;
    if (std::abs(gamma) >= std::abs(beta)) {
        y2 = TT(1);
        y3 = -wi / gamma;
    }
    else {
        y2 = -wi / beta;
        y3 = -TT(1);
    }

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

            for (idx_t j = 0; j < i; ++j) {
                v4_r[i] -= T44(j, i) * v4_r[j];
                v4_i[i] -= T44(j, i) * v4_i[j];
                v4_r[i + 1] -= T44(j, i + 1) * v4_r[j];
                v4_i[i + 1] -= T44(j, i + 1) * v4_i[j];
            }

            // Solve the complex 2x2 system:
            // y**H
            // *
            // [T44(i,i)- (wr + i*wi) T44(i,i+1)                 ]
            // [T44(i+1,  i)          T44(i+1,  i+1)- (wr + i*wi)]
            // =
            // [v4_r[i] + i*v4_i[i]       ]
            // [v4_r[i+1]   + i*v4_i[i+1] ]
            // Using real arithmetic only with Cramer's rule

            TT a11r = T44(i, i) - wr;
            TT a11i = wi;
            // a12 and a21 are switched to transpose the system
            TT a12 = T44(i + 1, i);
            TT a21 = T44(i, i + 1);
            TT a22r = T44(i + 1, i + 1) - wr;
            TT a22i = wi;

            TT b1r = v4_r[i];
            TT b1i = v4_i[i];
            TT b2r = v4_r[i + 1];
            TT b2i = v4_i[i + 1];

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

            v4_r[i] = x1r;
            v4_i[i] = x1i;
            v4_r[i + 1] = x2r;
            v4_i[i + 1] = x2i;

            i += 2;
        }
        else {
            // 1x1 block
            for (idx_t j = 0; j < i; ++j) {
                v4_r[i] -= T44(j, i) * v4_r[j];
                v4_i[i] -= T44(j, i) * v4_i[j];
            }

            // Do the complex division:
            // (v1_r[i] + i*v1_i[i]) / (T11(i, i) - (wr + i*wi))
            // in real arithmetic only
            TT a = v4_r[i];
            TT b = v4_i[i];
            TT c = T44(i, i) - wr;
            TT d = wi;
            TT denom = c * c + d * d;
            v4_r[i] = (a * c + b * d) / denom;
            v4_i[i] = (b * c - a * d) / denom;

            i += 1;
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC3_FORWARDSOLVE_HH
