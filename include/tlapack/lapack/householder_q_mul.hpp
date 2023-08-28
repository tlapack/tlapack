/// @file householder_q_mul.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HOUSEHOLDER_Q_MUL_HH
#define TLAPACK_HOUSEHOLDER_Q_MUL_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/unmq.hpp"
#include "tlapack/lapack/unmq_level2.hpp"

namespace tlapack {

/// @brief Variants of the algorithm to multiply by the unitary matrix Q,
/// defined by a set of Householder reflectors.
enum class HouseholderQMulVariant : char { Level2 = '2', Blocked = 'B' };

/// @brief Options struct for householder_q_mul()
struct HouseholderQMulOpts : public UnmqOpts {
    HouseholderQMulVariant variant = HouseholderQMulVariant::Blocked;
};

/**
 * @brief Workspace query of householder_q_mul()
 *
 * @param[in] side Specifies which side op(Q) is to be applied.
 *      - Side::Left:  C := op(Q) C;
 *      - Side::Right: C := C op(Q).
 *
 * @param[in] trans The operation $op(Q)$ to be used:
 *      - Op::NoTrans:      $op(Q) = Q$;
 *      - Op::ConjTrans:    $op(Q) = Q^H$.
 *      Op::Trans is a valid value if the data type of A is real. In this case,
 *      the algorithm treats Op::Trans as Op::ConjTrans.
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
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise:
 *       - if side = Side::Left,  the m-by-k matrix V;
 *       - if side = Side::Right, the n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:
 *       - if side = Side::Left,  the k-by-m matrix V;
 *       - if side = Side::Right, the k-by-n matrix V.

 * @param[in] tau Vector of length k.
 *      Scalar factors of the elementary reflectors.
 *
 * @param[in] C m-by-n matrix.
 *
 * @param[in] opts Options.
 *    - @c opts.variant: Variant of the algorithm to use.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
constexpr WorkInfo householder_q_mul_worksize(
    side_t side,
    trans_t trans,
    direction_t direction,
    storage_t storeMode,
    const matrixV_t& V,
    const vector_t& tau,
    const matrixC_t& C,
    const HouseholderQMulOpts& opts = {})
{
    if (opts.variant == HouseholderQMulVariant::Level2)
        return unmq_level2_worksize<T>(side, trans, direction, storeMode, V,
                                       tau, C);
    else
        return unmq_worksize<T>(side, trans, direction, storeMode, V, tau, C,
                                opts);
}

/** @copydoc householder_q_mul()
 *
 * Workspace is provided as an argument.
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_WORKSPACE work_t>
int householder_q_mul_work(side_t side,
                           trans_t trans,
                           direction_t direction,
                           storage_t storeMode,
                           const matrixV_t& V,
                           const vector_t& tau,
                           matrixC_t& C,
                           work_t& work,
                           const HouseholderQMulOpts& opts = {})
{
    if (opts.variant == HouseholderQMulVariant::Level2)
        return unmq_level2_work(side, trans, direction, storeMode, V, tau, C,
                                work);
    else
        return unmq_work(side, trans, direction, storeMode, V, tau, C, work,
                         opts);
}

/**
 * @brief Applies unitary matrix Q to a matrix C.
 *
 * - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 * - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 * - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 * - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @param[in] side Specifies which side op(Q) is to be applied.
 *      - Side::Left:  C := op(Q) C;
 *      - Side::Right: C := C op(Q).
 *
 * @param[in] trans The operation $op(Q)$ to be used:
 *      - Op::NoTrans:      $op(Q) = Q$;
 *      - Op::ConjTrans:    $op(Q) = Q^H$.
 *      Op::Trans is a valid value if the data type of A is real. In this case,
 *      the algorithm treats Op::Trans as Op::ConjTrans.
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
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise:
 *       - if side = Side::Left,  the m-by-k matrix V;
 *       - if side = Side::Right, the n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:
 *       - if side = Side::Left,  the k-by-m matrix V;
 *       - if side = Side::Right, the k-by-n matrix V.

 * @param[in] tau Vector of length k.
 *      Scalar factors of the elementary reflectors.
 *
 * @param[in,out] C m-by-n matrix.
 *      On exit, C is replaced by one of the following:
 *      - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 *      - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 *      - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 *      - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * @param[in] opts Options.
 *   - @c opts.variant: Variant of the algorithm to use.
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
template <TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int householder_q_mul(side_t side,
                      trans_t trans,
                      direction_t direction,
                      storage_t storeMode,
                      const matrixV_t& V,
                      const vector_t& tau,
                      matrixC_t& C,
                      const HouseholderQMulOpts& opts = {})
{
    if (opts.variant == HouseholderQMulVariant::Level2)
        return unmq_level2(side, trans, direction, storeMode, V, tau, C);
    else
        return unmq(side, trans, direction, storeMode, V, tau, C, opts);
}

}  // namespace tlapack

#endif