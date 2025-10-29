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
 *  y(T - w*I) = 0, where w is the k-th eigenvalue of T.
 *
 * This can be split into the block matrix:
 *                           (k-1)    1   (n-k)
 *  [y1  y2  y3] [ T11 - w*I  T12  T13      ]    [0] (k-1)
 *               [ 0          0    T23      ]  = [0] 1
 *               [ 0          0    T33 - w*I]    [0] (n-k)
 *
 * We choose y1 = 0
 *
 * The last block column then gives:
 * y3 * (T33 - w*I) = -y2 * T23
 *
 * If we choose y2 = 1, we can solve for y3 using forward substitution.
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
        v[i] = -T(k, i);
    }

    TT w = T(k, k);  // eigenvalue

    // Forward substitution to solve the system
    auto T33 = slice(T, range(k + 1, n), range(k + 1, n));
    auto y3 = slice(v, range(k + 1, n));

    if constexpr (is_complex<TT>) {
        // The matrix is complex, so there are no two-by-two blocks to
        // consider
        for (idx_t i = 0; i < size(y3); ++i) {
            for (idx_t j = 0; j < i; ++j) {
                y3[i] -= T33(j, i) * y3[j];
            }

            y3[i] = y3[i] / (T33(i, i) - w);
        }
    }
    else {
        // The matrix is real, so we need to consider potential
        // 2x2 blocks for complex conjugate eigenvalue pairs
        idx_t i = 0;
        while (i < size(y3)) {
            bool is_2x2_block = false;
            if (i + 1 < size(y3)) {
                if (T33(i + 1, i) != TT(0)) {
                    is_2x2_block = true;
                }
            }

            if (is_2x2_block) {
                // 2x2 block

                for (idx_t j = 0; j < i; ++j) {
                    y3[i] -= T33(j, i) * y3[j];
                    y3[i + 1] -= T33(j, i + 1) * y3[j];
                }

                // Solve the 2x2 (transposed) system:
                // [T33(i,i)-w   T33(i+1,i)    ] [y3[i]  ] = [rhs1]
                // [T33(i,i+1)   T33(i+1,i+1)-w] [y3[i+1]]   [rhs2]
                TT rhs1 = y3[i];
                TT rhs2 = y3[i + 1];

                TT a = T33(i, i) - w;
                TT b = T33(i + 1, i);
                TT c = T33(i, i + 1);
                TT d = T33(i + 1, i + 1) - w;

                TT det = a * d - b * c;

                y3[i] = (d * rhs1 - b * rhs2) / det;
                y3[i + 1] = (-c * rhs1 + a * rhs2) / det;

                i += 2;
            }
            else {
                // 1x1 block
                for (idx_t j = 0; j < i; ++j) {
                    y3[i] -= T33(j, i) * y3[j];
                }

                y3[i] = y3[i] / (T33(i, i) - w);

                i += 1;
            }
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC3_FORWARDSOLVE_HH
