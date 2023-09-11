/// @file rot_sequence.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zlasr.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ROT_SEQUENCE_HH
#define TLAPACK_ROT_SEQUENCE_HH

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
 * where P is an orthogonal matrix consisting of a sequence of k plane
 * rotations, with k = m-1 when side = Side::Left and k = n-1 when side =
 * Side::Right.
 *
 * When direction = Direction::Forward, then
 *
 *    P = P(k-1) * P(k-2) * ... * P(1) * P(0)
 *
 * and when direction = Direction::Backward, then
 *
 *    P = P(0) * P(1) * ... * P(k-2) * P(k-1),
 *
 * when P(i) is a plane rotation matrix defined as
 *
 *    P(i) = ( 1                                        )
 *           (    ...                                   )
 *           (         ...                              )
 *           (              c(i)        s(i)            )
 *           (             -conj(s(i))  c(i)            )
 *           (                                1         )
 *           (                                   ...    )
 *           (                                        1 )
 *
 *  which only modifies rows/columns i and i + 1.
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
 * @param[in] c Real vector of length k.
 *     Cosines of the rotations
 *
 * @param[in] s Vector of length k.
 *     Sines of the rotations
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @ingroup computational
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_SVECTOR C_t,
          TLAPACK_SVECTOR S_t,
          TLAPACK_SMATRIX A_t>
int rot_sequence(
    side_t side, direction_t direction, const C_t& c, const S_t& s, A_t& A)
{
    using T = type_t<A_t>;
    using idx_t = size_type<A_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = (side == Side::Left) ? m - 1 : n - 1;

    // quick return
    if (k < 1) return 0;

    if constexpr (layout<A_t> == Layout::ColMajor) {
        if (direction == Direction::Forward) {
            if (side == Side::Left) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i2 = k; i2 > 0; --i2) {
                        idx_t i = i2 - 1;
                        T temp = c[i] * A(i, j) + s[i] * A(i + 1, j);
                        A(i + 1, j) =
                            -conj(s[i]) * A(i, j) + c[i] * A(i + 1, j);
                        A(i, j) = temp;
                    }
                }
            }
            else {  // Side::Right
                // Manual unrolling of loop, applying 3 rotations at a time
                // This allows some parts of the vector to remain in register
                idx_t ii = k % 3;
                for (idx_t i2 = k; i2 > ii; i2 = i2 - 3) {
                    idx_t i = i2 - 1;

                    for (idx_t j = 0; j < m; ++j) {
                        T temp = A(j, i + 1);
                        T temp0 = A(j, i);
                        T temp1 = A(j, i - 1);

                        // Apply first rotation
                        A(j, i + 1) = -s[i] * temp0 + c[i] * temp;
                        temp0 = c[i] * temp0 + conj(s[i]) * temp;

                        // Apply second rotation
                        A(j, i) = -s[i - 1] * temp1 + c[i - 1] * temp0;
                        temp1 = c[i - 1] * temp1 + conj(s[i - 1]) * temp0;

                        // Apply third rotation
                        A(j, i - 1) =
                            -s[i - 2] * A(j, i - 2) + c[i - 2] * temp1;
                        A(j, i - 2) =
                            c[i - 2] * A(j, i - 2) + conj(s[i - 2]) * temp1;
                    }
                }
                // If the amount of rotations is not divisible by 3, apply the
                // final ones one by one
                for (idx_t i2 = ii; i2 > 0; --i2) {
                    idx_t i = i2 - 1;
                    for (idx_t j = 0; j < m; ++j) {
                        T temp = c[i] * A(j, i) + conj(s[i]) * A(j, i + 1);
                        A(j, i + 1) = -s[i] * A(j, i) + c[i] * A(j, i + 1);
                        A(j, i) = temp;
                    }
                }
            }
        }
        else {  // Direction::Backward
            if (side == Side::Left) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < k; ++i) {
                        T temp = c[i] * A(i, j) + s[i] * A(i + 1, j);
                        A(i + 1, j) =
                            -conj(s[i]) * A(i, j) + c[i] * A(i + 1, j);
                        A(i, j) = temp;
                    }
                }
            }
            else {  // Side::Right

                // Manual unrolling of loop, applying 3 rotations at a time
                // This allows some parts of the vector to remain in register
                idx_t ii = k - (k % 3);
                for (idx_t i = 0; i + 1 < ii; i = i + 3) {
                    for (idx_t j = 0; j < m; ++j) {
                        T temp = A(j, i);
                        T temp0 = A(j, i + 1);
                        T temp1 = A(j, i + 2);

                        // Apply first rotation
                        A(j, i) = c[i] * temp + conj(s[i]) * temp0;
                        temp0 = -s[i] * temp + c[i] * temp0;

                        // Apply second rotation
                        A(j, i + 1) = c[i + 1] * temp0 + conj(s[i + 1]) * temp1;
                        temp1 = -s[i + 1] * temp0 + c[i + 1] * temp1;

                        // Apply third rotation
                        A(j, i + 2) =
                            c[i + 2] * temp1 + conj(s[i + 2]) * A(j, i + 3);
                        A(j, i + 3) =
                            -s[i + 2] * temp1 + c[i + 2] * A(j, i + 3);
                    }
                }
                // If the amount of rotations is not divisible by 3, apply the
                // final ones one by one
                for (idx_t i = ii; i < k; ++i) {
                    for (idx_t j = 0; j < m; ++j) {
                        T temp = c[i] * A(j, i) + conj(s[i]) * A(j, i + 1);
                        A(j, i + 1) = -s[i] * A(j, i) + c[i] * A(j, i + 1);
                        A(j, i) = temp;
                    }
                }
            }
        }
    }
    else {
        if (direction == Direction::Forward) {
            if (side == Side::Left) {
                // Manual unrolling of loop, applying 3 rotations at a time
                // This allows some parts of the vector to remain in register
                idx_t ii = k % 3;
                for (idx_t i2 = k; i2 > ii; i2 = i2 - 3) {
                    idx_t i = i2 - 1;

                    for (idx_t j = 0; j < n; ++j) {
                        T temp = A(i + 1, j);
                        T temp0 = A(i, j);
                        T temp1 = A(i - 1, j);

                        // Apply first rotation
                        A(i + 1, j) = -conj(s[i]) * temp0 + c[i] * temp;
                        temp0 = c[i] * temp0 + s[i] * temp;

                        // Apply second rotation
                        A(i, j) = -conj(s[i - 1]) * temp1 + c[i - 1] * temp0;
                        temp1 = c[i - 1] * temp1 + s[i - 1] * temp0;

                        // Apply third rotation
                        A(i - 1, j) =
                            -conj(s[i - 2]) * A(i - 2, j) + c[i - 2] * temp1;
                        A(i - 2, j) = c[i - 2] * A(i - 2, j) + s[i - 2] * temp1;
                    }
                }
                // If the amount of rotations is not divisible by 3, apply the
                // final ones one by one
                for (idx_t i2 = ii; i2 > 0; --i2) {
                    idx_t i = i2 - 1;
                    for (idx_t j = 0; j < n; ++j) {
                        T temp = c[i] * A(i, j) + s[i] * A(i + 1, j);
                        A(i + 1, j) =
                            -conj(s[i]) * A(i, j) + c[i] * A(i + 1, j);
                        A(i, j) = temp;
                    }
                }
            }
            else {
                for (idx_t j = 0; j < m; ++j) {
                    for (idx_t i2 = k; i2 > 0; --i2) {
                        idx_t i = i2 - 1;
                        T temp = c[i] * A(j, i) + conj(s[i]) * A(j, i + 1);
                        A(j, i + 1) = -s[i] * A(j, i) + c[i] * A(j, i + 1);
                        A(j, i) = temp;
                    }
                }
            }
        }
        else {
            if (side == Side::Left) {
                // Manual unrolling of loop, applying 3 rotations at a time
                // This allows some parts of the vector to remain in register
                idx_t ii = k - (k % 3);
                for (idx_t i = 0; i + 1 < ii; i = i + 3) {
                    for (idx_t j = 0; j < n; ++j) {
                        T temp = A(i, j);
                        T temp0 = A(i + 1, j);
                        T temp1 = A(i + 2, j);

                        // Apply first rotation
                        A(i, j) = c[i] * temp + s[i] * temp0;
                        temp0 = -conj(s[i]) * temp + c[i] * temp0;

                        // Apply second rotation
                        A(i + 1, j) = c[i + 1] * temp0 + s[i + 1] * temp1;
                        temp1 = -conj(s[i + 1]) * temp0 + c[i + 1] * temp1;

                        // Apply third rotation
                        A(i + 2, j) = c[i + 2] * temp1 + s[i + 2] * A(i + 3, j);
                        A(i + 3, j) =
                            -conj(s[i + 2]) * temp1 + c[i + 2] * A(i + 3, j);
                    }
                }
                // If the amount of rotations is not divisible by 3, apply the
                // final ones one by one
                for (idx_t i = ii; i < k; ++i) {
                    for (idx_t j = 0; j < n; ++j) {
                        T temp = c[i] * A(i, j) + s[i] * A(i + 1, j);
                        A(i + 1, j) =
                            -conj(s[i]) * A(i, j) + c[i] * A(i + 1, j);
                        A(i, j) = temp;
                    }
                }
            }
            else {
                for (idx_t j = 0; j < m; ++j) {
                    for (idx_t i = 0; i < k; ++i) {
                        T temp = c[i] * A(j, i) + conj(s[i]) * A(j, i + 1);
                        A(j, i + 1) = -s[i] * A(j, i) + c[i] * A(j, i + 1);
                        A(j, i) = temp;
                    }
                }
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_ROT_SEQUENCE_HH
