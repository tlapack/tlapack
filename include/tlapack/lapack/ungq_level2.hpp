/// @file ungq_level2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zung2l.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGQ_LV2_HH
#define TLAPACK_UNGQ_LV2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/laset.hpp"

namespace tlapack {

/** Worspace query of ungq_level2()
 *
 * @param[in] direction
 *     Indicates how Q is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $op(Q) = H_1 H_2 ... H_k$.
 *     - Direction::Backward: $op(Q) = H_k ... H_2 H_1$.
 *     The operator $op(Q)$ is defined as
 *     - $op(Q) = Q$ if storeMode = StoreV::Columnwise, and
 *     - $op(Q) = Q^H$ if storeMode = StoreV::Rowwise.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] A m-by-n matrix.

 * @param[in] tau Vector of length k.
 *      Scalar factors of the elementary reflectors.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
inline constexpr WorkInfo ungq_level2_worksize(direction_t direction,
                                               storage_t storeMode,
                                               const matrix_t& A,
                                               const vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (storeMode == StoreV::Columnwise) {
        return (n > 1)
                   ? larf_worksize<T>(LEFT_SIDE, direction, COLUMNWISE_STORAGE,
                                      col(A, 0), tau[0], cols(A, range{1, n}))
                   : WorkInfo(0);
    }
    else {
        return (m > 1)
                   ? larf_worksize<T>(RIGHT_SIDE, direction, ROWWISE_STORAGE,
                                      row(A, 0), tau[0], rows(A, range{1, m}))
                   : WorkInfo(0);
    }
}

/**
 * @brief Generates a matrix Q that is the product of elementary reflectors.
 *
 * @param[in] direction
 *     Indicates how Q is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $op(Q) = H_1 H_2 ... H_k$.
 *     - Direction::Backward: $op(Q) = H_k ... H_2 H_1$.
 *     The operator $op(Q)$ is defined as
 *     - $op(Q) = Q$ if storeMode = StoreV::Columnwise, and
 *     - $op(Q) = Q^H$ if storeMode = StoreV::Rowwise.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *     See Further Details.
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry,
 *      - If storeMode = StoreV::Columnwise:
 *        - if direction = Direction::Forward, the m-by-k matrix V in the first
 *          k columns;
 *        - if direction = Direction::Backward, the m-by-k matrix V in the last
 *          k columns.
 *      - If storeMode = StoreV::Rowwise:
 *        - if direction = Direction::Forward, the k-by-n matrix V in the first
 *          k rows;
 *        - if direction = Direction::Backward, the k-by-n matrix V in the last
 *          k rows.
 *      On exit, the m-by-n matrix Q.
 *
 * @param[in] tau Vector of length k.
 *      Scalar factors of the elementary reflectors.
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @return 0 if success.
 *
 * @par Further Details
 *
 * The shape of the matrix V and the storage of the vectors which define the
 * $H_i$ is best illustrated by the following example with k = 3. The elements
 * equal to 1 are not accessed. The rest of the matrix is not used.
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
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_WORKSPACE work_t>
int ungq_level2_work(direction_t direction,
                     storage_t storeMode,
                     matrix_t& A,
                     const vector_t& tau,
                     work_t& work)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = size(tau);

    // check arguments
    tlapack_check((storeMode == StoreV::Columnwise) ? (m >= n && n >= k)
                                                    : (n >= m && m >= k));

    // quick return
    if (m <= 0 || n <= 0) return 0;

    if (storeMode == StoreV::Columnwise) {
        if (direction == Direction::Forward) {
            // Initialize unit matrix
            auto A0 = slice(A, range{0, k}, range{k, n});
            auto A1 = slice(A, range{k, m}, range{k, n});
            laset(GENERAL, zero, zero, A0);
            laset(GENERAL, zero, one, A1);

            for (idx_t j = k - 1; j != idx_t(-1); --j) {
                const T tauj = tau[j];

                // Apply $H_{j+1}$ to $A( j:m, j+1:n )$ from the left
                auto v = slice(A, range{j, m}, j);
                auto C = slice(A, range{j, m}, range{j + 1, n});
                larf_work(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, v, tauj, C,
                          work);

                // Update column j
                scal(-tauj, v);
                A(j, j) = one - tauj;
                for (idx_t i = 0; i < j; i++)
                    A(i, j) = zero;
            }
        }
        else {
            // Initialize unit matrix
            auto A0 = slice(A, range{0, m - n}, range{0, n - k});
            auto A1 = slice(A, range{m - n, m - k}, range{0, n - k});
            auto A2 = slice(A, range{m - k, m}, range{0, n - k});
            laset(GENERAL, zero, zero, A0);
            laset(GENERAL, zero, one, A1);
            laset(GENERAL, zero, zero, A2);

            for (idx_t j = 0; j < k; ++j) {
                const idx_t sizev = m - k + j + 1;
                const idx_t jj = n - k + j;
                const T tauj = tau[j];

                // Apply $H_{j+1}$ to $A( 0:sizev, 0:jj )$ from the left
                auto v = slice(A, range{0, sizev}, jj);
                auto C = slice(A, range{0, sizev}, range{0, jj});
                larf_work(LEFT_SIDE, BACKWARD, COLUMNWISE_STORAGE, v, tauj, C,
                          work);

                // Update column jj
                scal(-tauj, v);
                A(sizev - 1, jj) = one - tauj;
                for (idx_t i = sizev; i < m; i++)
                    A(i, jj) = zero;
            }
        }
    }
    else {
        if (direction == Direction::Forward) {
            // Initialize unit matrix
            auto A0 = slice(A, range{k, m}, range{0, k});
            auto A1 = slice(A, range{k, m}, range{k, n});
            laset(GENERAL, zero, zero, A0);
            laset(GENERAL, zero, one, A1);

            for (idx_t i = k - 1; i != idx_t(-1); --i) {
                const T taui = conj(tau[i]);

                // Apply $H_{i+1}$ to $A( i+1:m, i:n )$ from the right
                auto v = slice(A, i, range{i, n});
                auto C = slice(A, range{i + 1, m}, range{i, n});
                larf_work(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, v, taui, C,
                          work);

                // Update row i
                scal(-taui, v);
                A(i, i) = one - taui;
                for (idx_t j = 0; j < i; j++)
                    A(i, j) = zero;
            }
        }
        else {
            // Initialize unit matrix
            auto A0 = slice(A, range{0, m - k}, range{0, n - m});
            auto A1 = slice(A, range{0, m - k}, range{n - m, n - k});
            auto A2 = slice(A, range{0, m - k}, range{n - k, n});
            laset(GENERAL, zero, zero, A0);
            laset(GENERAL, zero, one, A1);
            laset(GENERAL, zero, zero, A2);

            for (idx_t i = 0; i < k; ++i) {
                const idx_t sizev = n - k + i + 1;
                const idx_t ii = m - k + i;
                const T taui = conj(tau[i]);

                // Apply $H_{i+1}$ to $A( 0:jj, 0:sizev )$ from the right
                auto v = slice(A, ii, range{0, sizev});
                auto C = slice(A, range{0, ii}, range{0, sizev});
                larf_work(RIGHT_SIDE, BACKWARD, ROWWISE_STORAGE, v, taui, C,
                          work);

                // Update row ii
                scal(-taui, v);
                A(ii, sizev - 1) = one - taui;
                for (idx_t j = sizev; j < n; j++)
                    A(ii, j) = zero;
            }
        }
    }

    return 0;
}

/**
 * @brief Generates a matrix Q that is the product of elementary reflectors.
 *
 * @param[in] direction
 *     Indicates how Q is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $op(Q) = H_1 H_2 ... H_k$.
 *     - Direction::Backward: $op(Q) = H_k ... H_2 H_1$.
 *     The operator $op(Q)$ is defined as
 *     - $op(Q) = Q$ if storeMode = StoreV::Columnwise, and
 *     - $op(Q) = Q^H$ if storeMode = StoreV::Rowwise.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *     @see ungq_level2_work().
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry,
 *      - If storeMode = StoreV::Columnwise:
 *        - if direction = Direction::Forward, the m-by-k matrix V in the first
 *          k columns;
 *        - if direction = Direction::Backward, the m-by-k matrix V in the last
 *          k columns.
 *      - If storeMode = StoreV::Rowwise:
 *        - if direction = Direction::Forward, the k-by-n matrix V in the first
 *          k rows;
 *        - if direction = Direction::Backward, the k-by-n matrix V in the last
 *          k rows.
 *      On exit, the m-by-n matrix Q.
 *
 * @param[in] tau Vector of length k.
 *      Scalar factors of the elementary reflectors.
 *
 * @return 0 if success.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int ungq_level2(direction_t direction,
                storage_t storeMode,
                matrix_t& A,
                const vector_t& tau)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // quick return
    if (m <= 0 || n <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = ungq_level2_worksize<T>(direction, storeMode, A, tau);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return ungq_level2_work(direction, storeMode, A, tau, work);
}

}  // namespace tlapack

#endif  // TLAPACK_UNGQ_LV2_HH
