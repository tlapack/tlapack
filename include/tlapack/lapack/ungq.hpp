/// @file ungq.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zungqr.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGQ_HH
#define TLAPACK_UNGQ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/ungq_level2.hpp"

namespace tlapack {

/**
 * Options struct for ungq
 */
struct UngqOpts {
    size_t nb = 32;  ///< Block size
};

/** Worspace query of ungq()
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
 * @param[in] opts Options.
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
inline constexpr WorkInfo ungq_worksize(direction_t direction,
                                        storage_t storeMode,
                                        const matrix_t& A,
                                        const vector_t& tau,
                                        const UngqOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using matrixT_t = matrix_type<matrix_t, vector_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = size(tau);
    const idx_t nb = min((idx_t)opts.nb, k);

    WorkInfo workinfo;
    const auto V = (storeMode == StoreV::Columnwise)
                       ? slice(A, range{0, m}, range{0, nb})
                       : slice(A, range{0, nb}, range{0, n});

    if (storeMode == StoreV::Columnwise) {
        // larfb:
        if (nb < n) {
            // Empty matrices
            const auto matrixT = slice(A, range{0, nb}, range{0, nb});
            const auto C = slice(A, range{0, m},
                                 (direction == Direction::Forward)
                                     ? range{nb, n}
                                     : range{0, (n - k) + ((k - 1) / nb) * nb});

            // Internal workspace queries
            workinfo = larfb_worksize<T>(LEFT_SIDE, NO_TRANS, direction,
                                         COLUMNWISE_STORAGE, V, matrixT, C);

            // Local workspace sizes
            if (is_same_v<T, type_t<matrixT_t>>) workinfo += WorkInfo(nb, nb);
        }
    }
    else {
        // larfb:
        if (nb < m) {
            // Empty matrices
            const auto matrixT = slice(A, range{0, nb}, range{0, nb});
            const auto C = slice(A,
                                 (direction == Direction::Forward)
                                     ? range{nb, m}
                                     : range{0, (m - k) + ((k - 1) / nb) * nb},
                                 range{0, n});

            // Internal workspace queries
            workinfo = larfb_worksize<T>(RIGHT_SIDE, CONJ_TRANS, direction,
                                         ROWWISE_STORAGE, V, matrixT, C);

            // Local workspace sizes
            if (is_same_v<T, type_t<matrixT_t>>) workinfo += WorkInfo(nb, nb);
        }
    }

    // ungq_level2:
    const auto taui = slice(tau, range{0, nb});
    workinfo.minMax(ungq_level2_worksize<T>(direction, storeMode, V, taui));

    return workinfo;
}

/**
 * @copydoc ungq_level2()
 *
 * Blocked algorithm.
 *
 * @param[in] opts Options.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int ungq(direction_t direction,
         storage_t storeMode,
         matrix_t& A,
         const vector_t& tau,
         const UngqOpts& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using matrixT_t = matrix_type<matrix_t, vector_t>;

    // Functor
    Create<matrixT_t> new_matrix;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = size(tau);
    const idx_t nb = min((idx_t)opts.nb, k);

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check((storeMode == StoreV::Columnwise) ? (m >= n && n >= k)
                                                    : (n >= m && m >= k));

    // quick return
    if (m <= 0 || n <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = ungq_worksize<T>(direction, storeMode, A, tau, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);
    auto matrixT = (min(m, n) > nb)
                       ? slice(work, range{workinfo.m - nb, workinfo.m},
                               range{workinfo.n - nb, workinfo.n})
                       : slice(work, range{0, 0}, range{0, 0});

    if (storeMode == StoreV::Columnwise) {
        if (direction == Direction::Forward) {
            // Initialize unit matrix
            auto A0 = slice(A, range{0, k}, range{k, n});
            auto A1 = slice(A, range{k, m}, range{k, n});
            laset(GENERAL, zero, zero, A0);
            laset(GENERAL, zero, one, A1);

            for (idx_t j = ((k - 1) / nb) * nb; j != -nb; j -= nb) {
                const idx_t ib = min(nb, k - j);
                const auto tauj = slice(tau, range{j, j + ib});

                auto V = slice(A, range{j, m}, range{j, j + ib});
                if (j + ib < n) {
                    // Apply block reflector to A( j:m, j+ib:n )$ from the left

                    auto matrixTj = slice(matrixT, range{0, ib}, range{0, ib});
                    auto C = slice(A, range{j, m}, range{j + ib, n});

                    larft(FORWARD, COLUMNWISE_STORAGE, V, tauj, matrixTj);
                    larfb_work(LEFT_SIDE, NO_TRANS, FORWARD, COLUMNWISE_STORAGE,
                               V, matrixTj, C, work);
                }

                // Apply block reflector to A( 0:m, j:j+ib )$ from the left
                // using unblocked code.
                auto A0 = slice(A, range{0, j}, range{j, j + ib});
                laset(GENERAL, zero, zero, A0);
                ungq_level2_work(FORWARD, COLUMNWISE_STORAGE, V, tauj, work);
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

            for (idx_t j = 0; j < k; j += nb) {
                const idx_t ib = min(nb, k - j);
                const idx_t sizev = m - k + j + ib;
                const idx_t jj = n - k + j;
                const auto tauj = slice(tau, range{j, j + ib});

                auto V = slice(A, range{0, sizev}, range{jj, jj + ib});
                if (jj > 0) {
                    // Apply block reflector to A( 0:sizev, 0:jj )$ from the
                    // left

                    auto matrixTj = slice(matrixT, range{0, ib}, range{0, ib});
                    auto C = slice(A, range{0, sizev}, range{0, jj});

                    larft(BACKWARD, COLUMNWISE_STORAGE, V, tauj, matrixTj);
                    larfb_work(LEFT_SIDE, NO_TRANS, BACKWARD,
                               COLUMNWISE_STORAGE, V, matrixTj, C, work);
                }

                // Apply block reflector to A( 0:m, jj:jj+ib )$ from the left
                // using unblocked code.
                auto A0 = slice(A, range{sizev, m}, range{jj, jj + ib});
                ungq_level2_work(BACKWARD, COLUMNWISE_STORAGE, V, tauj, work);
                laset(GENERAL, zero, zero, A0);
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

            for (idx_t i = ((k - 1) / nb) * nb; i != -nb; i -= nb) {
                const idx_t ib = min(nb, k - i);
                const auto taui = slice(tau, range{i, i + ib});

                auto V = slice(A, range{i, i + ib}, range{i, n});
                if (i + ib < m) {
                    // Apply block reflector to A( i+ib:m, i:n )$ from the right

                    auto matrixTi = slice(matrixT, range{0, ib}, range{0, ib});
                    auto C = slice(A, range{i + ib, m}, range{i, n});

                    larft(FORWARD, ROWWISE_STORAGE, V, taui, matrixTi);
                    larfb_work(RIGHT_SIDE, CONJ_TRANS, FORWARD, ROWWISE_STORAGE,
                               V, matrixTi, C, work);
                }

                // Apply block reflector to A( i:i+ib, 0:n )$ from the right
                // using unblocked code.
                auto A0 = slice(A, range{i, i + ib}, range{0, i});
                laset(GENERAL, zero, zero, A0);
                ungq_level2_work(FORWARD, ROWWISE_STORAGE, V, taui, work);
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

            for (idx_t i = 0; i < k; i += nb) {
                const idx_t ib = min(nb, k - i);
                const idx_t sizev = n - k + i + ib;
                const idx_t ii = m - k + i;
                const auto taui = slice(tau, range{i, i + ib});

                auto V = slice(A, range{ii, ii + ib}, range{0, sizev});
                if (ii > 0) {
                    // Apply block reflector to A( 0:ii, 0:sizev )$ from the
                    // left

                    auto matrixTi = slice(matrixT, range{0, ib}, range{0, ib});
                    auto C = slice(A, range{0, ii}, range{0, sizev});

                    larft(BACKWARD, ROWWISE_STORAGE, V, taui, matrixTi);
                    larfb_work(RIGHT_SIDE, CONJ_TRANS, BACKWARD,
                               ROWWISE_STORAGE, V, matrixTi, C, work);
                }

                // Apply block reflector to A( ii:ii+ib, 0:n )$ from the left
                // using unblocked code.
                auto A0 = slice(A, range{ii, ii + ib}, range{sizev, n});
                ungq_level2_work(BACKWARD, ROWWISE_STORAGE, V, taui, work);
                laset(GENERAL, zero, zero, A0);
            }
        }
    }

    return 0;
}  // namespace tlapack

}  // namespace tlapack

#endif  // TLAPACK_UNGQ_HH
