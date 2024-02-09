/// @file rot_sequence3.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ROT_SEQUENCE3_HH
#define TLAPACK_ROT_SEQUENCE3_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/lapack/rot_kernel.hpp"

namespace tlapack {

template <TLAPACK_SMATRIX C_t,
          TLAPACK_SMATRIX S_t,
          TLAPACK_SMATRIX A_t,
          typename idx_t>
void rot_sequence_forward_left(const idx_t m,
                               const idx_t n,
                               const idx_t k,
                               const C_t& C,
                               const S_t& S,
                               A_t& A)
{
    using T = type_t<A_t>;
    tlapack_check((idx_t)ncols(S) == k);
    tlapack_check((idx_t)ncols(C) == k);
    tlapack_check((idx_t)nrows(S) + 1 == m);
    tlapack_check((idx_t)nrows(C) + 1 == m);
    tlapack_check((idx_t)nrows(A) == m);
    tlapack_check((idx_t)ncols(A) == n);
    tlapack_check(m > k + 1);

    // Number of values that fit in a cache line
    // We assume cache lines are 64 bytes
    const idx_t nt = 64 / sizeof(T);
    // Leading dimension of A_pack, normally equal to n, but we make it a
    // multiple of nt so that each row of A_pack is aligned to a 64-byte
    // boundary.
    const idx_t ld_pack = ((n - 1 + nt) / nt) * nt;
    // Number of rows to pack
    const idx_t np = k + 2;
    // Array to store np aligned rows of A
    // Note: regardless of the layout of A, A_pack is always row-major
    // This is because we need the rows to be stored contiguously in memory
    // to be able to apply the rotations efficiently
    alignas(64) T A_pack[np * ld_pack];

    // Copy the first np rows of A to A_pack
    for (idx_t i = 0; i < np; ++i) {
        for (idx_t j = 0; j < n; ++j) {
            A_pack[i * ld_pack + j] = A(i, j);
        }
    }
    // i_pack points to the "first" row of A_pack
    idx_t i_pack = 0;

    // Startup phase, apply the upper left triangle of C and S
    for (idx_t j = 0; j + 1 < k; ++j) {
        for (idx_t i = 0, g = j; i < j + 1; ++i, --g) {
            rot_nofuse(n, &A_pack[g * ld_pack], &A_pack[(g + 1) * ld_pack],
                       C(g, i), S(g, i));
        }
    }

    // Pipeline phase
    for (idx_t j = k - 1; j + 1 < m - 1; j += 2) {
        for (idx_t i = 0, g = j, g2 = k - 1; i + 1 < k;
             i += 2, g -= 2, g2 -= 2) {
            T* a1 = &A_pack[((i_pack + g2 - 1) % np) * ld_pack];
            T* a2 = &A_pack[((i_pack + g2) % np) * ld_pack];
            T* a3 = &A_pack[((i_pack + g2 + 1) % np) * ld_pack];
            T* a4 = &A_pack[((i_pack + g2 + 2) % np) * ld_pack];

            rot_fuse2x2(n, a1, a2, a3, a4, C(g, i), S(g, i), C(g - 1, i + 1),
                        S(g - 1, i + 1), C(g + 1, i), S(g + 1, i), C(g, i + 1),
                        S(g, i + 1));
        }
        if (k % 2 == 1) {
            // k is odd, so there are two more rotations to apply
            idx_t i = k - 1;
            idx_t g = j - i;
            idx_t g2 = 1;

            T* a1 = &A_pack[((i_pack + g2 - 1) % np) * ld_pack];
            T* a2 = &A_pack[((i_pack + g2) % np) * ld_pack];
            T* a3 = &A_pack[((i_pack + g2 + 1) % np) * ld_pack];
            rot_fuse2x1(n, a1, a2, a3, C(g, i), S(g, i), C(g + 1, i),
                        S(g + 1, i));
        }
        // rows i_pack and i_pack+1 are finished, copy them back to A
        for (idx_t i = 0; i < n; i++) {
            A(j - (k - 1), i) = A_pack[i + i_pack * ld_pack];
            A(j + 1 - (k - 1), i) = A_pack[i + ((i_pack + 1) % np) * ld_pack];
        }
        // Pack next rows and update i_pack
        if (j + 4 < m) {
            for (idx_t i = 0; i < n; i++) {
                A_pack[i + i_pack * ld_pack] = A(j + 3, i);
                A_pack[i + ((i_pack + 1) % np) * ld_pack] = A(j + 4, i);
            }
            i_pack = (i_pack + 2) % np;
        }
        else if (j + 3 < m) {
            // Load new row
            for (idx_t i = 0; i < n; i++) {
                A_pack[i + i_pack * ld_pack] = A(j + 3, i);
            }
            i_pack = (i_pack + 1) % np;
        }
    }

    // Shutdown phase
    for (idx_t j = (m - k + 1) % 2; j < k; ++j) {
        for (idx_t i = j, g = m - 2, g2 = k - 1; i < k; ++i, --g, --g2) {
            T* a1 = &A_pack[((i_pack + g2 + 1) % np) * ld_pack];
            T* a2 = &A_pack[((i_pack + g2 + 2) % np) * ld_pack];
            rot_nofuse(n, a1, a2, C(g, i), S(g, i));
        }
        // Row i_pack + j+1 in the packed matrix is finished, store it
        for (idx_t i = 0; i < n; i++) {
            A(m - k - 1 + j, i) = A_pack[i + ((i_pack + j + 1) % np) * ld_pack];
        }
    }

    // Store the last column of the packed matrix
    for (idx_t i = 0; i < n; i++) {
        A(m - 1, i) = A_pack[i + ((i_pack + k + 1) % np) * ld_pack];
    }
}

template <TLAPACK_SMATRIX C_t,
          TLAPACK_SMATRIX S_t,
          TLAPACK_SMATRIX A_t,
          typename idx_t>
void rot_sequence_forward_right(const idx_t m,
                                const idx_t n,
                                const idx_t k,
                                const C_t& C,
                                const S_t& S,
                                A_t& A)
{
    using T = type_t<A_t>;
    tlapack_check((idx_t)ncols(S) == k);
    tlapack_check((idx_t)ncols(C) == k);
    tlapack_check((idx_t)nrows(S) + 1 == n);
    tlapack_check((idx_t)nrows(C) + 1 == n);
    tlapack_check((idx_t)nrows(A) == m);
    tlapack_check((idx_t)ncols(A) == n);
    tlapack_check(n > k + 1);

    // Number of values that fit in a cache line
    // We assume cache lines are 64 bytes
    const idx_t nt = 64 / sizeof(T);
    // Leading dimension of A_pack, normally equal to m, but we make it a
    // multiple of nt so that each row of A_pack is aligned to a 64-byte
    // boundary.
    const idx_t ld_pack = ((m - 1 + nt) / nt) * nt;
    // Number of columns to pack
    const idx_t np = k + 2;
    // Array to store np aligned columns of A
    // Note: regardless of the layout of A, A_pack is always column-major
    // This is because we need the columns to be stored contiguously in memory
    // to be able to apply the rotations efficiently
    alignas(64) T A_pack[np * ld_pack];

    // Copy the first np columns of A to A_pack
    for (idx_t j = 0; j < np; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A_pack[i + j * ld_pack] = A(i, j);
        }
    }
    // i_pack points to the "first" column of A_pack
    idx_t i_pack = 0;

    // Startup phase, apply the upper left triangle of C and S
    for (idx_t j = 0; j + 1 < k; ++j) {
        for (idx_t i = 0, g = j; i < j + 1; ++i, --g) {
            rot_nofuse(m, &A_pack[g * ld_pack], &A_pack[(g + 1) * ld_pack],
                       C(g, i), conj(S(g, i)));
        }
    }

    // Pipeline phase
    for (idx_t j = k - 1; j + 1 < n - 1; j += 2) {
        for (idx_t i = 0, g = j, g2 = k - 1; i + 1 < k;
             i += 2, g -= 2, g2 -= 2) {
            T* a1 = &A_pack[((i_pack + g2 - 1) % np) * ld_pack];
            T* a2 = &A_pack[((i_pack + g2) % np) * ld_pack];
            T* a3 = &A_pack[((i_pack + g2 + 1) % np) * ld_pack];
            T* a4 = &A_pack[((i_pack + g2 + 2) % np) * ld_pack];

            rot_fuse2x2(m, a1, a2, a3, a4, C(g, i), conj(S(g, i)),
                        C(g - 1, i + 1), conj(S(g - 1, i + 1)), C(g + 1, i),
                        conj(S(g + 1, i)), C(g, i + 1), conj(S(g, i + 1)));
        }
        if (k % 2 == 1) {
            // k is odd, so there are two more rotations to apply
            idx_t i = k - 1;
            idx_t g = j - i;
            idx_t g2 = 1;

            T* a1 = &A_pack[((i_pack + g2 - 1) % np) * ld_pack];
            T* a2 = &A_pack[((i_pack + g2) % np) * ld_pack];
            T* a3 = &A_pack[((i_pack + g2 + 1) % np) * ld_pack];
            rot_fuse2x1(m, a1, a2, a3, C(g, i), conj(S(g, i)), C(g + 1, i),
                        conj(S(g + 1, i)));
        }
        // columns i_pack and i_pack+1 are finished, copy them back to A
        for (idx_t i = 0; i < m; i++) {
            A(i, j - (k - 1)) = A_pack[i + i_pack * ld_pack];
            A(i, j + 1 - (k - 1)) = A_pack[i + ((i_pack + 1) % np) * ld_pack];
        }
        // Pack next columns and update i_pack
        if (j + 4 < n) {
            for (idx_t i = 0; i < m; i++) {
                A_pack[i + i_pack * ld_pack] = A(i, j + 3);
                A_pack[i + ((i_pack + 1) % np) * ld_pack] = A(i, j + 4);
            }
            i_pack = (i_pack + 2) % np;
        }
        else if (j + 3 < n) {
            // Load new row
            for (idx_t i = 0; i < m; i++) {
                A_pack[i + i_pack * ld_pack] = A(i, j + 3);
            }
            i_pack = (i_pack + 1) % np;
        }
    }

    // Shutdown phase
    for (idx_t j = (n - k + 1) % 2; j < k; ++j) {
        for (idx_t i = j, g = n - 2, g2 = k - 1; i < k; ++i, --g, --g2) {
            T* a1 = &A_pack[((i_pack + g2 + 1) % np) * ld_pack];
            T* a2 = &A_pack[((i_pack + g2 + 2) % np) * ld_pack];
            rot_nofuse(m, a1, a2, C(g, i), conj(S(g, i)));
        }
        // Row i_pack + j+1 in the packed matrix is finished, store it
        for (idx_t i = 0; i < m; i++) {
            A(i, n - k - 1 + j) = A_pack[i + ((i_pack + j + 1) % np) * ld_pack];
        }
    }

    // Store the last column of the packed matrix
    for (idx_t i = 0; i < m; i++) {
        A(i, n - 1) = A_pack[i + ((i_pack + k + 1) % np) * ld_pack];
    }
}

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
 * Reference: "Restructuring the Tridiagonal and Bidiagonal QR algorithms for
 * Performance" F. G. Van Zee, R. A. Van de Geijn, G. Quintana-Orti
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
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = (side == Side::Left) ? m - 1 : n - 1;
    const idx_t l = ncols(C);

    // Blocking parameter
    const idx_t nb = 256;
    const idx_t nb_l = 64;

    // Check dimensions
    tlapack_check((idx_t)ncols(S) == l);
    tlapack_check((idx_t)nrows(C) == k);
    tlapack_check((idx_t)nrows(S) == k);
    // tlapack_check(k >= l);

    // quick return
    if (k < 1 or l < 1) return;

    // If the dimensions don't allow for effective pipelining,
    // then apply the rotations using rot_sequence
    if (l == 1 or k < l + 1) {
        for (idx_t j = 0; j < l; ++j) {
            auto c = col(C, j);
            auto s = col(S, j);
            rot_sequence(side, direction, c, s, A);
        }
        return;
    }

    if (direction == Direction::Forward) {
        if (side == Side::Left) {
            for (idx_t ib = 0; ib < n; ib += nb) {
                idx_t ib2 = std::min(ib + nb, n);
                auto A2 = cols(A, range(ib, ib2));
                for (idx_t il = 0; il < l; il += nb_l) {
                    idx_t il2 = std::min(il + nb_l, l);
                    auto C2 = cols(C, range(il, il2));
                    auto S2 = cols(S, range(il, il2));
                    rot_sequence_forward_left(m, ib2 - ib, il2 - il, C2, S2,
                                              A2);
                }
            }
            return;
        }
        else {
            for (idx_t ib = 0; ib < m; ib += nb) {
                idx_t ib2 = std::min(ib + nb, m);
                auto A2 = rows(A, range(ib, ib2));
                for (idx_t il = 0; il < l; il += nb_l) {
                    idx_t il2 = std::min(il + nb_l, l);
                    auto C2 = cols(C, range(il, il2));
                    auto S2 = cols(S, range(il, il2));
                    rot_sequence_forward_right(ib2 - ib, n, il2 - il, C2, S2,
                                               A2);
                }
            }
            return;
        }
    }

    // Apply rotations
    if constexpr (layout<A_t> == Layout::ColMajor) {
        if (side == Side::Left) {
            if (direction == Direction::Backward) {
                // Left side, backward direction
                for (idx_t ib = 0; ib < n; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, n);
                    // Startup phase
                    for (idx_t i1 = 0; i1 < n; ++i1) {
                        for (idx_t j = 0; j < l - 1; ++j) {
                            for (idx_t i = 0, g2 = j; i < j + 1; ++i, --g2) {
                                idx_t g = m - 2 - g2;
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = l - 1; j + 1 < m - 1; j += 2) {
                            for (idx_t i = 0, g2 = j; i + 1 < l;
                                 i += 2, g2 -= 2) {
                                idx_t g = m - 2 - g2;
                                //
                                // Apply first rotation
                                //

                                // A(g,i1) after first rotation
                                T temp1 =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                // A(g+1,i1) after first rotation
                                T temp2 = -conj(S(g, i)) * A(g, i1) +
                                          C(g, i) * A(g + 1, i1);

                                //
                                // Apply second rotation
                                //

                                // A(g,i1) after second rotation
                                T temp3 = -conj(S(g - 1, i)) * A(g - 1, i1) +
                                          C(g - 1, i) * temp1;
                                A(g - 1, i1) = C(g - 1, i) * A(g - 1, i1) +
                                               S(g - 1, i) * temp1;

                                //
                                // Apply third rotation
                                //

                                // A(g+1,i1) after third rotation
                                T temp4 = C(g + 1, i + 1) * temp2 +
                                          S(g + 1, i + 1) * A(g + 2, i1);
                                A(g + 2, i1) = -conj(S(g + 1, i + 1)) * temp2 +
                                               C(g + 1, i + 1) * A(g + 2, i1);

                                // Apply fourth rotation
                                A(g, i1) =
                                    C(g, i + 1) * temp3 + S(g, i + 1) * temp4;
                                A(g + 1, i1) = -conj(S(g, i + 1)) * temp3 +
                                               C(g, i + 1) * temp4;
                            }

                            if (l % 2 == 1) {
                                // Apply two more rotations that could not be
                                // fused
                                idx_t i = l - 1;
                                idx_t g2 = j - (l - 1);
                                idx_t g = m - 2 - g2;

                                // Apply first rotation
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;

                                // Apply second rotation
                                T temp2 = C(g - 1, i) * A(g - 1, i1) +
                                          S(g - 1, i) * A(g, i1);
                                A(g, i1) = -conj(S(g - 1, i)) * A(g - 1, i1) +
                                           C(g - 1, i) * A(g, i1);
                                A(g - 1, i1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = ((m - l + 1) % 2); j < l; ++j) {
                            for (idx_t i = j, g2 = m - 2; i < l; ++i, --g2) {
                                idx_t g = m - 2 - g2;
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                }
            }
            else {
                // Left side, backward direction
                for (idx_t ib = 0; ib < n; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, n);
                    // Startup phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = 0; j < l - 1; ++j) {
                            for (idx_t i = 0, g = j; i < j + 1; ++i, --g) {
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = l - 1; j + 1 < m - 1; j += 2) {
                            for (idx_t i = 0, g = j; i + 1 < l;
                                 i += 2, g -= 2) {
                                //
                                // Apply first rotation
                                //

                                // A(g,i1) after first rotation
                                T temp1 =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                // A(g+1,i1) after first rotation
                                T temp2 = -conj(S(g, i)) * A(g, i1) +
                                          C(g, i) * A(g + 1, i1);

                                //
                                // Apply second rotation
                                //

                                // A(g+1,i1) after second rotation
                                T temp3 = C(g + 1, i) * temp2 +
                                          S(g + 1, i) * A(g + 2, i1);
                                A(g + 2, i1) = -conj(S(g + 1, i)) * temp2 +
                                               C(g + 1, i) * A(g + 2, i1);

                                //
                                // Apply third rotation
                                //

                                // A(g,i1) after third rotation
                                T temp4 =
                                    -conj(S(g - 1, i + 1)) * A(g - 1, i1) +
                                    C(g - 1, i + 1) * temp1;
                                A(g - 1, i1) = C(g - 1, i + 1) * A(g - 1, i1) +
                                               S(g - 1, i + 1) * temp1;

                                // Apply fourth rotation
                                A(g, i1) =
                                    C(g, i + 1) * temp4 + S(g, i + 1) * temp3;
                                A(g + 1, i1) = -conj(S(g, i + 1)) * temp4 +
                                               C(g, i + 1) * temp3;
                            }

                            if (l % 2 == 1) {
                                // Apply two more rotations that could not be
                                // fused
                                idx_t i = l - 1;
                                idx_t g = j - (l - 1);

                                // Apply first rotation
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;

                                // Apply second rotation
                                T temp2 = C(g + 1, i) * A(g + 1, i1) +
                                          S(g + 1, i) * A(g + 2, i1);
                                A(g + 2, i1) =
                                    -conj(S(g + 1, i)) * A(g + 1, i1) +
                                    C(g + 1, i) * A(g + 2, i1);
                                A(g + 1, i1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = ((m - l + 1) % 2); j < l; ++j) {
                            for (idx_t i = j, g = m - 2; i < l; ++i, --g) {
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                }
            }
        }
        else {
            if (direction == Direction::Backward) {
                // Right side, forward direction
                for (idx_t ib = 0; ib < m; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, m);
                    // Startup phase
                    for (idx_t j = 0; j < l - 1; ++j) {
                        for (idx_t i = 0, g2 = j; i < j + 1; ++i, --g2) {
                            idx_t g = n - 2 - g2;
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t j = l - 1; j + 1 < n - 1; j += 2) {
                        for (idx_t i = 0, g2 = j; i + 1 < l; i += 2, g2 -= 2) {
                            idx_t g = n - 2 - g2;
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                //
                                // Apply first rotation
                                //

                                // A(i1,g) after first rotation
                                T temp1 = C(g, i) * A(i1, g) +
                                          conj(S(g, i)) * A(i1, g + 1);
                                // A(i1,g+1) after first rotation
                                T temp2 = -S(g, i) * A(i1, g) +
                                          C(g, i) * A(i1, g + 1);

                                //
                                // Apply second rotation
                                //

                                // A(i1,g) after second rotation
                                T temp3 = -S(g - 1, i) * A(i1, g - 1) +
                                          C(g - 1, i) * temp1;
                                A(i1, g - 1) = C(g - 1, i) * A(i1, g - 1) +
                                               conj(S(g - 1, i)) * temp1;
                                //
                                // Apply third rotation
                                //

                                // A(i1,g+1) after third rotation
                                T temp4 = C(g + 1, i + 1) * temp2 +
                                          conj(S(g + 1, i + 1)) * A(i1, g + 2);
                                A(i1, g + 2) = -S(g + 1, i + 1) * temp2 +
                                               C(g + 1, i + 1) * A(i1, g + 2);

                                //
                                // Apply fourth rotation
                                //

                                A(i1, g) = C(g, i + 1) * temp3 +
                                           conj(S(g, i + 1)) * temp4;
                                A(i1, g + 1) =
                                    -S(g, i + 1) * temp3 + C(g, i + 1) * temp4;
                            }
                        }
                        if (l % 2 == 1) {
                            // Apply two more rotations that could not be fused
                            idx_t i = l - 1;
                            idx_t g2 = j - (l - 1);
                            idx_t g = n - 2 - g2;

                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                // Apply first rotation
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;

                                // Apply second rotation
                                T temp2 = C(g - 1, i) * A(i1, g - 1) +
                                          conj(S(g - 1, i)) * A(i1, g);
                                A(i1, g) = -S(g - 1, i) * A(i1, g - 1) +
                                           C(g - 1, i) * A(i1, g);
                                A(i1, g - 1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t j = ((n - l + 1) % 2); j < l; ++j) {
                        for (idx_t i = j, g2 = n - 2; i < l; ++i, --g2) {
                            idx_t g = n - 2 - g2;
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                }
            }
            else {
                // Right side, backward direction
                for (idx_t ib = 0; ib < m; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, m);
                    // Startup phase
                    for (idx_t j = 0; j < l - 1; ++j) {
                        for (idx_t i = 0, g = j; i < j + 1; ++i, --g) {
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t j = l - 1; j + 1 < n - 1; j += 2) {
                        for (idx_t i = 0, g = j; i + 1 < l; i += 2, g -= 2) {
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                //
                                // Apply first rotation
                                //

                                // A(i1,g) after first rotation
                                T temp1 = C(g, i) * A(i1, g) +
                                          conj(S(g, i)) * A(i1, g + 1);
                                // A(i1,g+1) after first rotation
                                T temp2 = -S(g, i) * A(i1, g) +
                                          C(g, i) * A(i1, g + 1);

                                //
                                // Apply second rotation
                                //

                                // A(i1,g+1) after second rotation
                                T temp3 = C(g + 1, i) * temp2 +
                                          conj(S(g + 1, i)) * A(i1, g + 2);
                                A(i1, g + 2) = -S(g + 1, i) * temp2 +
                                               C(g + 1, i) * A(i1, g + 2);

                                //
                                // Apply third rotation
                                //

                                // A(i1,g) after third rotation
                                T temp4 = -S(g - 1, i + 1) * A(i1, g - 1) +
                                          C(g - 1, i + 1) * temp1;
                                A(i1, g - 1) = C(g - 1, i + 1) * A(i1, g - 1) +
                                               conj(S(g - 1, i + 1)) * temp1;

                                // Apply fourth rotation
                                A(i1, g) = C(g, i + 1) * temp4 +
                                           conj(S(g, i + 1)) * temp3;
                                A(i1, g + 1) =
                                    -S(g, i + 1) * temp4 + C(g, i + 1) * temp3;
                            }
                        }
                        if (l % 2 == 1) {
                            // Apply two more rotations that could not be fused
                            idx_t i = l - 1;
                            idx_t g = j - (l - 1);

                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                // Apply first rotation
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;

                                // Apply second rotation
                                T temp2 = C(g + 1, i) * A(i1, g + 1) +
                                          conj(S(g + 1, i)) * A(i1, g + 2);
                                A(i1, g + 2) = -S(g + 1, i) * A(i1, g + 1) +
                                               C(g + 1, i) * A(i1, g + 2);
                                A(i1, g + 1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t j = ((n - l + 1) % 2); j < l; ++j) {
                        for (idx_t i = j, g = n - 2; i < l; ++i, --g) {
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        if (side == Side::Left) {
            if (direction == Direction::Backward) {
                // Left side, forward direction
                for (idx_t ib = 0; ib < n; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, n);
                    // Startup phase
                    for (idx_t j = 0; j < l - 1; ++j) {
                        for (idx_t i = 0, g2 = j; i < j + 1; ++i, --g2) {
                            idx_t g = m - 2 - g2;
                            for (idx_t i1 = 0; i1 < n; ++i1) {
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t j = l - 1; j + 1 < m - 1; j += 2) {
                        for (idx_t i = 0, g2 = j; i + 1 < l; i += 2, g2 -= 2) {
                            idx_t g = m - 2 - g2;
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                //
                                // Apply first rotation
                                //

                                // A(g,i1) after first rotation
                                T temp1 =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                // A(g+1,i1) after first rotation
                                T temp2 = -conj(S(g, i)) * A(g, i1) +
                                          C(g, i) * A(g + 1, i1);

                                //
                                // Apply second rotation
                                //

                                // A(g,i1) after second rotation
                                T temp3 = -conj(S(g - 1, i)) * A(g - 1, i1) +
                                          C(g - 1, i) * temp1;
                                A(g - 1, i1) = C(g - 1, i) * A(g - 1, i1) +
                                               S(g - 1, i) * temp1;

                                //
                                // Apply third rotation
                                //

                                // A(g+1,i1) after third rotation
                                T temp4 = C(g + 1, i + 1) * temp2 +
                                          S(g + 1, i + 1) * A(g + 2, i1);
                                A(g + 2, i1) = -conj(S(g + 1, i + 1)) * temp2 +
                                               C(g + 1, i + 1) * A(g + 2, i1);

                                // Apply fourth rotation
                                A(g, i1) =
                                    C(g, i + 1) * temp3 + S(g, i + 1) * temp4;
                                A(g + 1, i1) = -conj(S(g, i + 1)) * temp3 +
                                               C(g, i + 1) * temp4;
                            }
                        }

                        if (l % 2 == 1) {
                            // Apply two more rotations that could not be
                            // fused
                            idx_t i = l - 1;
                            idx_t g2 = j - (l - 1);
                            idx_t g = m - 2 - g2;

                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                // Apply first rotation
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;

                                // Apply second rotation
                                T temp2 = C(g - 1, i) * A(g - 1, i1) +
                                          S(g - 1, i) * A(g, i1);
                                A(g, i1) = -conj(S(g - 1, i)) * A(g - 1, i1) +
                                           C(g - 1, i) * A(g, i1);
                                A(g - 1, i1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t j = ((m - l + 1) % 2); j < l; ++j) {
                        for (idx_t i = j, g2 = m - 2; i < l; ++i, --g2) {
                            idx_t g = m - 2 - g2;
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                }
            }
            else {
                // Left side, backward direction
                for (idx_t ib = 0; ib < n; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, n);
                    // Startup phase
                    for (idx_t j = 0; j < l - 1; ++j) {
                        for (idx_t i = 0, g = j; i < j + 1; ++i, --g) {
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t j = l - 1; j + 1 < m - 1; j += 2) {
                        for (idx_t i = 0, g = j; i + 1 < l; i += 2, g -= 2) {
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                //
                                // Apply first rotation
                                //

                                // A(g,i1) after first rotation
                                T temp1 =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                // A(g+1,i1) after first rotation
                                T temp2 = -conj(S(g, i)) * A(g, i1) +
                                          C(g, i) * A(g + 1, i1);

                                //
                                // Apply second rotation
                                //

                                // A(g+1,i1) after second rotation
                                T temp3 = C(g + 1, i) * temp2 +
                                          S(g + 1, i) * A(g + 2, i1);
                                A(g + 2, i1) = -conj(S(g + 1, i)) * temp2 +
                                               C(g + 1, i) * A(g + 2, i1);

                                //
                                // Apply third rotation
                                //

                                // A(g,i1) after third rotation
                                T temp4 =
                                    -conj(S(g - 1, i + 1)) * A(g - 1, i1) +
                                    C(g - 1, i + 1) * temp1;
                                A(g - 1, i1) = C(g - 1, i + 1) * A(g - 1, i1) +
                                               S(g - 1, i + 1) * temp1;

                                // Apply fourth rotation
                                A(g, i1) =
                                    C(g, i + 1) * temp4 + S(g, i + 1) * temp3;
                                A(g + 1, i1) = -conj(S(g, i + 1)) * temp4 +
                                               C(g, i + 1) * temp3;
                            }
                        }

                        if (l % 2 == 1) {
                            // Apply two more rotations that could not be
                            // fused
                            idx_t i = l - 1;
                            idx_t g = j - (l - 1);

                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                // Apply first rotation
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;

                                // Apply second rotation
                                T temp2 = C(g + 1, i) * A(g + 1, i1) +
                                          S(g + 1, i) * A(g + 2, i1);
                                A(g + 2, i1) =
                                    -conj(S(g + 1, i)) * A(g + 1, i1) +
                                    C(g + 1, i) * A(g + 2, i1);
                                A(g + 1, i1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t j = ((m - l + 1) % 2); j < l; ++j) {
                        for (idx_t i = j, g = m - 2; i < l; ++i, --g) {
                            for (idx_t i1 = ib; i1 < ib2; ++i1) {
                                T temp =
                                    C(g, i) * A(g, i1) + S(g, i) * A(g + 1, i1);
                                A(g + 1, i1) = -conj(S(g, i)) * A(g, i1) +
                                               C(g, i) * A(g + 1, i1);
                                A(g, i1) = temp;
                            }
                        }
                    }
                }
            }
        }
        else {
            if (direction == Direction::Backward) {
                // Right side, forward direction
                for (idx_t ib = 0; ib < m; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, m);
                    // Startup phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = 0; j < l - 1; ++j) {
                            for (idx_t i = 0, g2 = j; i < j + 1; ++i, --g2) {
                                idx_t g = n - 2 - g2;
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = l - 1; j + 1 < n - 1; j += 2) {
                            for (idx_t i = 0, g2 = j; i + 1 < l;
                                 i += 2, g2 -= 2) {
                                idx_t g = n - 2 - g2;
                                //
                                // Apply first rotation
                                //

                                // A(i1,g) after first rotation
                                T temp1 = C(g, i) * A(i1, g) +
                                          conj(S(g, i)) * A(i1, g + 1);
                                // A(i1,g+1) after first rotation
                                T temp2 = -S(g, i) * A(i1, g) +
                                          C(g, i) * A(i1, g + 1);

                                //
                                // Apply second rotation
                                //

                                // A(i1,g) after second rotation
                                T temp3 = -S(g - 1, i) * A(i1, g - 1) +
                                          C(g - 1, i) * temp1;
                                A(i1, g - 1) = C(g - 1, i) * A(i1, g - 1) +
                                               conj(S(g - 1, i)) * temp1;
                                //
                                // Apply third rotation
                                //

                                // A(i1,g+1) after third rotation
                                T temp4 = C(g + 1, i + 1) * temp2 +
                                          conj(S(g + 1, i + 1)) * A(i1, g + 2);
                                A(i1, g + 2) = -S(g + 1, i + 1) * temp2 +
                                               C(g + 1, i + 1) * A(i1, g + 2);

                                //
                                // Apply fourth rotation
                                //

                                A(i1, g) = C(g, i + 1) * temp3 +
                                           conj(S(g, i + 1)) * temp4;
                                A(i1, g + 1) =
                                    -S(g, i + 1) * temp3 + C(g, i + 1) * temp4;
                            }
                            if (l % 2 == 1) {
                                // Apply two more rotations that could not be
                                // fused
                                idx_t i = l - 1;
                                idx_t g2 = j - (l - 1);
                                idx_t g = n - 2 - g2;

                                // Apply first rotation
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;

                                // Apply second rotation
                                T temp2 = C(g - 1, i) * A(i1, g - 1) +
                                          conj(S(g - 1, i)) * A(i1, g);
                                A(i1, g) = -S(g - 1, i) * A(i1, g - 1) +
                                           C(g - 1, i) * A(i1, g);
                                A(i1, g - 1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = ((n - l + 1) % 2); j < l; ++j) {
                            for (idx_t i = j, g2 = n - 2; i < l; ++i, --g2) {
                                idx_t g = n - 2 - g2;
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                }
            }
            else {
                // Right side, backward direction
                for (idx_t ib = 0; ib < m; ib += nb) {
                    idx_t ib2 = std::min(ib + nb, m);
                    // Startup phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = 0; j < l - 1; ++j) {
                            for (idx_t i = 0, g = j; i < j + 1; ++i, --g) {
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                    // Pipeline phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = l - 1; j + 1 < n - 1; j += 2) {
                            for (idx_t i = 0, g = j; i + 1 < l;
                                 i += 2, g -= 2) {
                                //
                                // Apply first rotation
                                //

                                // A(i1,g) after first rotation
                                T temp1 = C(g, i) * A(i1, g) +
                                          conj(S(g, i)) * A(i1, g + 1);
                                // A(i1,g+1) after first rotation
                                T temp2 = -S(g, i) * A(i1, g) +
                                          C(g, i) * A(i1, g + 1);

                                //
                                // Apply second rotation
                                //

                                // A(i1,g+1) after second rotation
                                T temp3 = C(g + 1, i) * temp2 +
                                          conj(S(g + 1, i)) * A(i1, g + 2);
                                A(i1, g + 2) = -S(g + 1, i) * temp2 +
                                               C(g + 1, i) * A(i1, g + 2);

                                //
                                // Apply third rotation
                                //

                                // A(i1,g) after third rotation
                                T temp4 = -S(g - 1, i + 1) * A(i1, g - 1) +
                                          C(g - 1, i + 1) * temp1;
                                A(i1, g - 1) = C(g - 1, i + 1) * A(i1, g - 1) +
                                               conj(S(g - 1, i + 1)) * temp1;

                                // Apply fourth rotation
                                A(i1, g) = C(g, i + 1) * temp4 +
                                           conj(S(g, i + 1)) * temp3;
                                A(i1, g + 1) =
                                    -S(g, i + 1) * temp4 + C(g, i + 1) * temp3;
                            }
                            if (l % 2 == 1) {
                                // Apply two more rotations that could not be
                                // fused
                                idx_t i = l - 1;
                                idx_t g = j - (l - 1);

                                // Apply first rotation
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;

                                // Apply second rotation
                                T temp2 = C(g + 1, i) * A(i1, g + 1) +
                                          conj(S(g + 1, i)) * A(i1, g + 2);
                                A(i1, g + 2) = -S(g + 1, i) * A(i1, g + 1) +
                                               C(g + 1, i) * A(i1, g + 2);
                                A(i1, g + 1) = temp2;
                            }
                        }
                    }
                    // Shutdown phase
                    for (idx_t i1 = ib; i1 < ib2; ++i1) {
                        for (idx_t j = ((n - l + 1) % 2); j < l; ++j) {
                            for (idx_t i = j, g = n - 2; i < l; ++i, --g) {
                                T temp = C(g, i) * A(i1, g) +
                                         conj(S(g, i)) * A(i1, g + 1);
                                A(i1, g + 1) = -S(g, i) * A(i1, g) +
                                               C(g, i) * A(i1, g + 1);
                                A(i1, g) = temp;
                            }
                        }
                    }
                }
            }
        }
    }
}  // namespace tlapack

}  // namespace tlapack

#endif  // TLAPACK_ROT_SEQUENCE3_HH
