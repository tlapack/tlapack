/// @file rot_sequence3.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ROT_SEQUENCE3_HH
#define TLAPACK_ROT_SEQUENCE3_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"

namespace tlapack {

/** Applies a sequence of plane rotations to an (m-by-n) matrix
 *
 * When side = Side::Left, the transformation takes the form
 *
 *     A := P*A
 *
 * and when side = Side::Right, the transformation takes the form
 *
 *     A := A*P**H
 *
 * where P is an orthogonal matrix consisting of a sequence of k*l plane
 * rotations, with k = m-1 when side = Side::Left and k = n-1 when side =
 * Side::Right and l is the number of columns of C and S.
 *
 * When direction = Direction::Forward, then
 *
 *    P = P(k-1,l-1) * ... * P(0,l-1) * ... * P(k-1,l-2) * ... * P(0,l-2)
 *        * ... * P(k-1,0) * ... * P(0,0)
 *
 * and when direction = Direction::Backward, then
 *
 *    P = P(0,l-1) * ... * P(k-1,l-1) * ... * P(0,l-2) * ... * P(k-1,l-2)
 *        * ... * P(0,0) * ... * P(k-1,0)
 *
 * where P(i,j) is a plane rotation matrix defined as
 *
 *    P(i,j) = ( 1                                            )
 *             (    ...                                       )
 *             (         ...                                  )
 *             (              C(i,j)        S(i,j)            )
 *             (             -conj(S(i,j))  C(i,j)            )
 *             (                                    1         )
 *             (                                       ...    )
 *             (                                            1 )
 *
 *  which only modifies rows/columns i and i + 1.
 *
 * This function is a variant of rot_sequence, which also applies sequence of
 * plane rotations to a matrix. This variant can been seen as multiple
 * applications of rot_sequence, but more cache-efficient.
 *
 * Note: One of the implicit blocking parameters in this routine is l.
 * The routine will work blocks in A of size 2*l-by-nb, where nb is a blocking
 * parameter. If l is large, it may be better to call this routine multiple
 * times.
 *
 * @return  0 if success
 *
 * @param[in] side
 *      Specifies whether the plane rotation matrix P is applied to A
 *      on the left or the right
 *      - Side::Left:  A := P * A;
 *      - Side::Right: A := A * P**T.
 *
 * @param[in] direction
 *     Specifies whether P is a forward or backward sequence of plane rotations.
 *
 * @param[in] C Real k-by-l matrix.
 *     Cosines of the rotations
 *
 * @param[in] S k-by-l matrix.
 *     Sines of the rotations
 *
 * @param[in,out] A m-by-n matrix.
 *     Matrix that the rotations will be applied to.
 *
 * @ingroup computational
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_SMATRIX C_t,
          TLAPACK_SMATRIX S_t,
          TLAPACK_SMATRIX A_t>
void rot_sequence3(
    side_t side, direction_t direction, const C_t& C, const S_t& S, A_t& A)
{
    using T = type_t<A_t>;
    using idx_t = size_type<A_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = (side == Side::Left) ? m - 1 : n - 1;
    const idx_t l = ncols(C);

    // Blocking parameter
    const idx_t nb = 256;

    // Check dimensions
    tlapack_check((idx_t)ncols(S) == l);
    tlapack_check((idx_t)nrows(C) == k);
    tlapack_check((idx_t)nrows(S) == k);
    // tlapack_check(k >= l);

    // quick return
    if (k < 1 or l < 1) return;

    // If there is only one sequence, then use rot_sequence
    if (l == 1) {
        auto c = col(C, 0);
        auto s = col(S, 0);
        rot_sequence(side, direction, c, s, A);
        return;
    }

    // Apply rotations
    if (side == Side::Left) {
        if (direction == Direction::Forward) {
            // Number of blocks
            const idx_t n_blocks = (k + l - 1) / l + 1;

            // Apply the rotations in blocks
            for (idx_t jb = 0; jb < n; jb += nb) {
                for (idx_t b = n_blocks; b > 0; --b) {
                    for (idx_t h = 0; h < l; h++) {
                        for (idx_t i2 = std::min((b - 1) * l + h, k);
                             i2 > std::max<idx_t>((b - 1) * l + h, l) - l;
                             --i2) {
                            idx_t i = i2 - 1;
                            for (idx_t j = jb; j < std::min<idx_t>(n, jb + nb);
                                 ++j) {
                                T temp =
                                    C(i, h) * A(i, j) + S(i, h) * A(i + 1, j);
                                A(i + 1, j) = -conj(S(i, h)) * A(i, j) +
                                              C(i, h) * A(i + 1, j);
                                A(i, j) = temp;
                            }
                        }
                    }
                }
            }
        }
        else {
            // Number of blocks
            const idx_t n_blocks = (k + l - 1) / l + 1;

            // Apply the rotations in blocks
            for (idx_t jb = 0; jb < n; jb += nb) {
                for (idx_t b = 1; b <= n_blocks; ++b) {
                    for (idx_t h = 0; h < l; h++) {
                        for (idx_t i = std::max<idx_t>(b * l - h, l) - l;
                             i < std::min(b * l - h, k); ++i) {
                            for (idx_t j = jb; j < std::min<idx_t>(n, jb + nb);
                                 ++j) {
                                T temp =
                                    C(i, h) * A(i, j) + S(i, h) * A(i + 1, j);
                                A(i + 1, j) = -conj(S(i, h)) * A(i, j) +
                                              C(i, h) * A(i + 1, j);
                                A(i, j) = temp;
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        if (direction == Direction::Forward) {
            // Number of blocks
            const idx_t n_blocks = (k + l - 1) / l + 1;

            // Apply the rotations in blocks
            for (idx_t jb = 0; jb < m; jb += nb) {
                for (idx_t b = n_blocks; b > 0; --b) {
                    for (idx_t h = 0; h < l; h++) {
                        for (idx_t i2 = std::min((b - 1) * l + h, k);
                             i2 > std::max<idx_t>((b - 1) * l + h, l) - l;
                             --i2) {
                            idx_t i = i2 - 1;
                            for (idx_t j = jb; j < std::min<idx_t>(m, jb + nb);
                                 ++j) {
                                T temp = C(i, h) * A(j, i) +
                                         conj(S(i, h)) * A(j, i + 1);
                                A(j, i + 1) =
                                    -S(i, h) * A(j, i) + C(i, h) * A(j, i + 1);
                                A(j, i) = temp;
                            }
                        }
                    }
                }
            }
        }
        else {
            // Number of blocks
            const idx_t n_blocks = (k + l - 1) / l + 1;

            // Apply the rotations in blocks
            for (idx_t jb = 0; jb < m; jb += nb) {
                for (idx_t b = 1; b <= n_blocks; ++b) {
                    for (idx_t h = 0; h < l; h++) {
                        for (idx_t i = std::max<idx_t>(b * l - h, l) - l;
                             i < std::min(b * l - h, k); ++i) {
                            for (idx_t j = jb; j < std::min<idx_t>(m, jb + nb);
                                 ++j) {
                                T temp = C(i, h) * A(j, i) +
                                         conj(S(i, h)) * A(j, i + 1);
                                A(j, i + 1) =
                                    -S(i, h) * A(j, i) + C(i, h) * A(j, i + 1);
                                A(j, i) = temp;
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_ROT_SEQUENCE3_HH
