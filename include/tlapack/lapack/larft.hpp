/// @file larft.hpp Forms the triangular factor T of a block reflector.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larft.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARFT_HH
#define TLAPACK_LARFT_HH

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
 * @param[in] tau Vector of length k containing the scalar factors
 *      of the elementary reflectors H.
 *
 * @param[out] T Matrix of size k-by-k containing the triangular factors
 *      of the block reflector.
 *     - Direction::Forward:  T is upper triangular.
 *     - Direction::Backward: T is lower triangular.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SMATRIX matrixV_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SMATRIX matrixT_t>
int larft(direction_t direction,
          storage_t storeMode,
          const matrixV_t& V,
          const vector_t& tau,
          matrixT_t& T)
{
    // data traits
    using scalar_t = type_t<matrixT_t>;
    using real_t = real_type<scalar_t>;
    using tau_t = type_t<vector_t>;
    using idx_t = size_type<matrixV_t>;

    // using
    using pair = pair<idx_t, idx_t>;

    // constants
    const real_t one(1);
    const real_t zero(0);
    const idx_t n = (storeMode == StoreV::Columnwise) ? nrows(V) : ncols(V);
    const idx_t k = size(tau);

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check_false(storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise);
    tlapack_check_false(
        k > ((storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V)));
    tlapack_check_false(nrows(T) < k || ncols(T) < k);

    // Quick return
    if (n == 0 || k == 0) return 0;

    if (direction == Direction::Forward) {
        // First iteration:
        T(0, 0) = tau[0];

        // Remaining iterations:
        for (idx_t i = 1; i < k; ++i) {
            // Column vector t := T(0:i,i)
            auto t = slice(T, pair{0, i}, i);

            if (tau[i] == tau_t(0)) {
                // H(i) =  I
                for (idx_t j = 0; j < i; ++j)
                    t[j] = zero;
            }

            else {
                // General case
                if (storeMode == StoreV::Columnwise) {
                    // t := - tau[i] conj( V(i,0:i) )
                    for (idx_t j = 0; j < i; ++j)
                        t[j] = -tau[i] * conj(V(i, j));

                    // t := t - tau[i] V(i+1:n,0:i)^H V(i+1:n,i)
                    if (i + 1 < n) {
                        gemv(conjTranspose, -tau[i],
                             slice(V, pair{i + 1, n}, pair{0, i}),
                             slice(V, pair{i + 1, n}, i), one, t);
                    }
                }
                else {
                    // t := - tau[i] V(0:i,i)
                    for (idx_t j = 0; j < i; ++j)
                        t[j] = -tau[i] * V(j, i);

                    // t := t - tau[i] V(0:i,i:n) V(i,i+1:n)^H
                    if (i + 1 < n) {
                        auto Ti = slice(T, pair{0, i}, pair{i, i + 1});
                        gemm(noTranspose, conjTranspose, -tau[i],
                             slice(V, pair{0, i}, pair{i + 1, n}),
                             slice(V, pair{i, i + 1}, pair{i + 1, n}), one, Ti);
                    }
                }

                // t := T(0:i,0:i) * t
                trmv(upperTriangle, noTranspose, nonUnit_diagonal,
                     slice(T, pair{0, i}, pair{0, i}), t);
            }

            // Update diagonal
            T(i, i) = tau[i];
        }
    }
    else  // direct==Direction::Backward
    {
        // First iteration:
        T(k - 1, k - 1) = tau[k - 1];

        // Remaining iterations:
        for (idx_t i = k - 2; i != idx_t(-1); --i) {
            // Column vector t := T(0:i,i)
            auto t = slice(T, pair{i + 1, k}, i);

            if (tau[i] == tau_t(0)) {
                // H(i) =  I
                for (idx_t j = 0; j < k - i - 1; ++j)
                    t[j] = zero;
            }

            else {
                // General case
                if (storeMode == StoreV::Columnwise) {
                    // t := - tau[i] conj(V(n-k+i,i+1:k))
                    for (idx_t j = 0; j < k - i - 1; ++j)
                        t[j] = -tau[i] * conj(V(n - k + i, j + i + 1));

                    // t := t - tau[i] V(0:n-k+i,i+1:k)^H V(0:n-k+i,i)
                    gemv(conjTranspose, -tau[i],
                         slice(V, pair{0, n - k + i}, pair{i + 1, k}),
                         slice(V, pair{0, n - k + i}, i), one, t);
                }
                else {
                    // t := - tau[i] V(i+1:k,n-k+i)
                    for (idx_t j = 0; j < k - i - 1; ++j)
                        t[j] = -tau[i] * V(j + i + 1, n - k + i);

                    // t := t - tau[i] V(i+1:k,0:n-k+i) V(i,0:n-k+i)^H
                    auto Ti = slice(T, pair{i + 1, k}, pair{i, i + 1});
                    gemm(noTranspose, conjTranspose, -tau[i],
                         slice(V, pair{i + 1, k}, pair{0, n - k + i}),
                         slice(V, pair{i, i + 1}, pair{0, n - k + i}), one, Ti);
                }

                // t := T(i+1:k,i+1:k) * t
                trmv(lowerTriangle, noTranspose, nonUnit_diagonal,
                     slice(T, pair{i + 1, k}, pair{i + 1, k}), t);
            }

            // Update diagonal
            T(i, i) = tau[i];
        }
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LARFT_HH
