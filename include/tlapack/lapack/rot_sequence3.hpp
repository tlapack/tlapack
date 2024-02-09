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

    // If the dimensions don't allow for effective pipelining,
    // then apply the rotations using rot_sequence
    if (l == 1 or k < l) {
        for (idx_t j = 0; j < l; ++j) {
            auto c = col(C, j);
            auto s = col(S, j);
            rot_sequence(side, direction, c, s, A);
        }
        return;
    }

    // Apply rotations
    if constexpr (layout<A_t> == Layout::ColMajor) {
        if (side == Side::Left) {
            if (direction == Direction::Backward) {
                // Left side, forward direction
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
