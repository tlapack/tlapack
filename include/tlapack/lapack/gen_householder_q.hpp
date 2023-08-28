/// @file gen_householder_q.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEN_HOUSEHOLDER_Q_HH
#define TLAPACK_GEN_HOUSEHOLDER_Q_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ungq.hpp"
#include "tlapack/lapack/ungq_level2.hpp"

namespace tlapack {

/// @brief Variants of the algorithm to generate the unitary matrix Q from
/// a set of Householder reflectors.
enum class GenHouseholderQVariant : char { Level2 = '2', Blocked = 'B' };

/// @brief Options struct for gen_householder_q()
struct GenHouseholderQOpts : public UngqOpts {
    GenHouseholderQVariant variant = GenHouseholderQVariant::Blocked;
};

/** Worspace query of gen_householder_q()
 *
 * @param[in] direction Direction of the Householder reflectors.
 *
 * @param[in] storeMode Storage mode of the Householder reflectors.
 *
 * @param[in] A m-by-n matrix.
 *
 * @param[in] tau vector of length k.
 *
 * @param[in] opts Options.
 *     - @c opts.variant: Variant of the algorithm to use.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
constexpr WorkInfo gen_householder_q_worksize(
    direction_t direction,
    storage_t storeMode,
    const matrix_t& A,
    const vector_t& tau,
    const GenHouseholderQOpts& opts = {})
{
    if (opts.variant == GenHouseholderQVariant::Level2)
        return ungq_level2_worksize<T>(direction, storeMode, A, tau);
    else
        return ungq_worksize<T>(direction, storeMode, A, tau, opts);
}

/**
 * @copydoc gen_householder_q()
 *
 * Workspace is provided as an argument.
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_WORKSPACE work_t>
int gen_householder_q_work(direction_t direction,
                           storage_t storeMode,
                           matrix_t& A,
                           const vector_t& tau,
                           work_t& work,
                           const GenHouseholderQOpts& opts = {})
{
    if (opts.variant == GenHouseholderQVariant::Level2)
        return ungq_level2_work(direction, storeMode, A, tau, work);
    else
        return ungq_work(direction, storeMode, A, tau, work, opts);
}

/**
 * @brief Generates a matrix Q that is the product of elementary reflectors.
 *
 * @param[in] direction
 *     Indicates how Q is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $Q = H_1 H_2 ... H_k$.
 *     - Direction::Backward: $Q = H_k ... H_2 H_1$.
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
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
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
 * @ingroup variant_interface
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int gen_householder_q(direction_t direction,
                      storage_t storeMode,
                      matrix_t& A,
                      const vector_t& tau,
                      const GenHouseholderQOpts& opts = {})
{
    if (opts.variant == GenHouseholderQVariant::Level2)
        return ungq_level2(direction, storeMode, A, tau);
    else
        return ungq(direction, storeMode, A, tau, opts);
}

}  // namespace tlapack

#endif