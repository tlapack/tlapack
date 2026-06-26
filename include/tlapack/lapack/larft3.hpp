/// @file larft3.hpp Forms the triangular factor T of a block reflector.
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larft.h
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARFT3_HH
#define TLAPACK_LARFT3_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/trmv.hpp"

namespace tlapack {

/** Forms the triangular factor T of a block reflector H of order n,
 * which is defined as a product of k elementary reflectors.
 *
 * If direction = Direction::Forward,  H = H_1 H_2 ... H_k and T is upper
 * triangular. If direction = Direction::Backward, H = H_k ... H_2 H_1 and T is
 * lower triangular.
 *
 * If storeMode = StoreV::Columnwise, the vector which defines the elementary
 * reflector H(i) is stored in the i-th column of the array V, and
 *
 *               H  =  I - V * T * V'
 *
 * The shape of the matrix V and the storage of the vectors which define
 * the H(i) is best illustrated by the following example with n = 5 and
 * k = 3. The elements equal to 1 are not stored. The rest of the
 * array is not used.
 *
 *     direction = Forward and          direction = Forward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *         ( v1  1    )                     (     1 v2 v2 v2 )
 *         ( v1 v2  1 )                     (        1 v3 v3 )
 *         ( v1 v2 v3 )
 *         ( v1 v2 v3 )
 *
 *     direction = Backward and         direction = Backward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
 *         ( v1 v2 v3 )                     ( v2 v2 v2  1    )
 *         (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
 *         (     1 v3 )
 *         (        1 )
 *
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise: n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:    k-by-n matrix V.
 *
 * @param[out] Tmatrix Matrix of size k-by-k containing the triangular factors
 *      of the block reflector.
 *     - Direction::Forward:  T is upper triangular.
 *     - Direction::Backward: T is lower triangular.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_MATRIX matrix_a,
          TLAPACK_MATRIX matrix_h>
int larft3(direction_t direction,
           storage_t storeMode,
           const matrix_a& V,
           matrix_h& Tmatrix)
{
    // data traits
    using std::size_t;
    using idx_t = size_type<matrix_a>;
    using T = type_t<matrix_a>;

    // using
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = (storeMode == StoreV::Columnwise) ? nrows(V) : ncols(V);
    const idx_t k = (storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V);

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check_false(storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise);
    tlapack_check_false(
        k > ((storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V)));
    tlapack_check_false(nrows(Tmatrix) < k || ncols(Tmatrix) < k);

    // Quick return
    if (n == 0 || k == 0) return 0;

    if (direction == Direction::Forward) {
        for (idx_t i = 1; i < k; ++i) {
            // Column vector t := T(0:i,i)
            auto t = slice(Tmatrix, range{0, i}, i);
            T tau = Tmatrix(i, i);

            if (Tmatrix(i, i) == T(0.0)) {
                // H(i) =  I
                for (idx_t j = 0; j < i; ++j)
                    t[j] = T(0.0);
            }

            else {
                // General case
                if (storeMode == StoreV::Columnwise) {
                    // t := - tau conj( V(i,0:i) )
                    for (idx_t j = 0; j < i; ++j)
                        t[j] = -tau * conj(V(i, j));

                    // t := t - tau V(i+1:n,0:i)^H V(i+1:n,i)
                    if (i + 1 < n) {
                        gemv(CONJ_TRANS, -tau,
                             slice(V, range{i + 1, n}, range{0, i}),
                             slice(V, range{i + 1, n}, i), T(1.0), t);
                    }
                }
                else {
                    // t := - tau V(0:i,i)
                    for (idx_t j = 0; j < i; ++j)
                        t[j] = -tau * V(j, i);

                    // t := t - tau V(0:i,i:n) V(i,i+1:n)^H
                    if (i + 1 < n) {
                        auto Ti = slice(Tmatrix, range{0, i}, range{i, i + 1});
                        gemm(NO_TRANS, CONJ_TRANS, -tau,
                             slice(V, range{0, i}, range{i + 1, n}),
                             slice(V, range{i, i + 1}, range{i + 1, n}), T(1.0),
                             Ti);
                    }
                }

                // t := T(0:i,0:i) * t
                trmv(UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                     slice(Tmatrix, range{0, i}, range{0, i}), t);
            }

            // Update diagonal
            Tmatrix(i, i) = tau;
        }
    }
    else  // direct==Direction::Backward
    {
        // Remaining iterations:
        for (idx_t i = k - 2; i != idx_t(-1); --i) {
            // Column vector t := T(0:i,i)
            auto t = slice(Tmatrix, range{i + 1, k}, i);
            T tau = Tmatrix(i, i);

            if (tau == T(0.0)) {
                // H(i) =  I
                for (idx_t j = 0; j < k - i - 1; ++j)
                    t[j] = T(0.0);
            }

            else {
                // General case
                if (storeMode == StoreV::Columnwise) {
                    // t := - tau conj(V(n-k+i,i+1:k))
                    for (idx_t j = 0; j < k - i - 1; ++j)
                        t[j] = -tau * conj(V(n - k + i, j + i + 1));

                    // t := t - tau V(0:n-k+i,i+1:k)^H V(0:n-k+i,i)
                    gemv(CONJ_TRANS, -tau,
                         slice(V, range{0, n - k + i}, range{i + 1, k}),
                         slice(V, range{0, n - k + i}, i), T(1.0), t);
                }
                else {
                    // t := - tau V(i+1:k,n-k+i)
                    for (idx_t j = 0; j < k - i - 1; ++j)
                        t[j] = -tau * V(j + i + 1, n - k + i);

                    // t := t - tau[i] V(i+1:k,0:n-k+i) V(i,0:n-k+i)^H
                    auto Ti = slice(Tmatrix, range{i + 1, k}, range{i, i + 1});
                    gemm(NO_TRANS, CONJ_TRANS, -tau,
                         slice(V, range{i + 1, k}, range{0, n - k + i}),
                         slice(V, range{i, i + 1}, range{0, n - k + i}), T(1.0),
                         Ti);
                }

                // t := T(i+1:k,i+1:k) * t
                trmv(LOWER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                     slice(Tmatrix, range{i + 1, k}, range{i + 1, k}), t);
            }

            // Update diagonal
            Tmatrix(i, i) = tau;
        }
    }
    return 0;
}

/**
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise: n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:    k-by-n matrix V.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_MATRIX matrix_a>
int larft3(direction_t direction, storage_t storeMode, matrix_a& V)
{
    // data traits
    using std::size_t;
    using idx_t = size_type<matrix_a>;
    using T = type_t<matrix_a>;

    // using
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = (storeMode == StoreV::Columnwise) ? nrows(V) : ncols(V);
    const idx_t k = (storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V);

    // Upper Triangle of V, which holds the T matrix
    auto Tmatrix = slice(V, range{0, k}, range{0, k});

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check_false(storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise);
    tlapack_check_false(
        k > ((storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V)));
    tlapack_check_false(nrows(Tmatrix) < k || ncols(Tmatrix) < k);

    // Quick return
    if (n == 0 || k == 0) return 0;

    if (direction == Direction::Forward) {
        for (idx_t i = 1; i < k; ++i) {
            // Column vector t := T(0:i,i)
            auto t = slice(Tmatrix, range{0, i}, i);
            T tau = Tmatrix(i, i);

            if (Tmatrix(i, i) == T(0.0)) {
                // H(i) =  I
                for (idx_t j = 0; j < i; ++j)
                    t[j] = T(0.0);
            }

            else {
                // General case
                if (storeMode == StoreV::Columnwise) {
                    // t := - tau conj( V(i,0:i) )
                    for (idx_t j = 0; j < i; ++j)
                        t[j] = -tau * conj(V(i, j));

                    // t := t - tau V(i+1:n,0:i)^H V(i+1:n,i)
                    if (i + 1 < n) {
                        gemv(CONJ_TRANS, -tau,
                             slice(V, range{i + 1, n}, range{0, i}),
                             slice(V, range{i + 1, n}, i), T(1.0), t);
                    }
                }
                else {
                    // t := - tau V(0:i,i)
                    for (idx_t j = 0; j < i; ++j)
                        t[j] = -tau * V(j, i);

                    // t := t - tau V(0:i,i:n) V(i,i+1:n)^H
                    if (i + 1 < n) {
                        auto Ti = slice(Tmatrix, range{0, i}, range{i, i + 1});
                        gemm(NO_TRANS, CONJ_TRANS, -tau,
                             slice(V, range{0, i}, range{i + 1, n}),
                             slice(V, range{i, i + 1}, range{i + 1, n}), T(1.0),
                             Ti);
                    }
                }

                // t := T(0:i,0:i) * t
                trmv(UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                     slice(Tmatrix, range{0, i}, range{0, i}), t);
            }

            // Update diagonal
            Tmatrix(i, i) = tau;
        }
    }
    else  // direct==Direction::Backward
    {
        // Remaining iterations:
        for (idx_t i = k - 2; i != idx_t(-1); --i) {
            // Column vector t := T(0:i,i)
            auto t = slice(Tmatrix, range{i + 1, k}, i);
            T tau = Tmatrix(i, i);

            if (tau == T(0.0)) {
                // H(i) =  I
                for (idx_t j = 0; j < k - i - 1; ++j)
                    t[j] = T(0.0);
            }

            else {
                // General case
                if (storeMode == StoreV::Columnwise) {
                    // t := - tau conj(V(n-k+i,i+1:k))
                    for (idx_t j = 0; j < k - i - 1; ++j)
                        t[j] = -tau * conj(V(n - k + i, j + i + 1));

                    // t := t - tau V(0:n-k+i,i+1:k)^H V(0:n-k+i,i)
                    gemv(CONJ_TRANS, -tau,
                         slice(V, range{0, n - k + i}, range{i + 1, k}),
                         slice(V, range{0, n - k + i}, i), T(1.0), t);
                }
                else {
                    // t := - tau V(i+1:k,n-k+i)
                    for (idx_t j = 0; j < k - i - 1; ++j)
                        t[j] = -tau * V(j + i + 1, n - k + i);

                    // t := t - tau[i] V(i+1:k,0:n-k+i) V(i,0:n-k+i)^H
                    auto Ti = slice(Tmatrix, range{i + 1, k}, range{i, i + 1});
                    gemm(NO_TRANS, CONJ_TRANS, -tau,
                         slice(V, range{i + 1, k}, range{0, n - k + i}),
                         slice(V, range{i, i + 1}, range{0, n - k + i}), T(1.0),
                         Ti);
                }

                // t := T(i+1:k,i+1:k) * t
                trmv(LOWER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                     slice(Tmatrix, range{i + 1, k}, range{i + 1, k}), t);
            }

            // Update diagonal
            Tmatrix(i, i) = tau;
        }
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LARFT3_HH
