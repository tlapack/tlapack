/// @file unmq_level2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zunm2r.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNMQ_LV2_HH
#define TLAPACK_UNMQ_LV2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Worspace query of unmq_level2()
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
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
constexpr WorkInfo unmq_level2_worksize(side_t side,
                                        trans_t trans,
                                        direction_t direction,
                                        storage_t storeMode,
                                        const matrixV_t& V,
                                        const vector_t& tau,
                                        const matrixC_t& C)
{
    using idx_t = size_type<matrixC_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t nQ = (side == Side::Left) ? m : n;

    if (storeMode == StoreV::Columnwise)
        return larf_worksize<T>(side, direction, storeMode,
                                slice(V, range{0, nQ}, 0), tau[0], C);
    else
        return larf_worksize<T>(side, direction, storeMode,
                                slice(V, 0, range{0, nQ}), tau[0], C);
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
template <TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_WORKSPACE work_t>
int unmq_level2_work(side_t side,
                     trans_t trans,
                     direction_t direction,
                     storage_t storeMode,
                     const matrixV_t& V,
                     const vector_t& tau,
                     matrixC_t& C,
                     work_t& work)
{
    using idx_t = size_type<matrixC_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nQ = (side == Side::Left) ? m : n;

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(
        trans != Op::NoTrans && trans != Op::ConjTrans &&
        ((trans != Op::Trans) || is_complex<type_t<matrixV_t>>));
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check((storeMode == StoreV::Columnwise)
                      ? ((ncols(V) == k) && (nrows(V) == nQ))
                      : ((nrows(V) == k) && (ncols(V) == nQ)));

    // quick return
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    // const expressions
    const bool positiveIncLeft =
        (storeMode == StoreV::Columnwise)
            ? ((direction == Direction::Backward) ? (trans == Op::NoTrans)
                                                  : (trans != Op::NoTrans))
            : ((direction == Direction::Forward) ? (trans == Op::NoTrans)
                                                 : (trans != Op::NoTrans));
    const bool positiveInc =
        (side == Side::Left) ? positiveIncLeft : !positiveIncLeft;
    const idx_t i0 = (positiveInc) ? 0 : k - 1;
    const idx_t iN = (positiveInc) ? k : -1;
    const idx_t inc = (positiveInc) ? 1 : -1;

    if (storeMode == StoreV::Columnwise) {
        if (side == Side::Left) {
            for (idx_t i = i0; i != iN; i += inc) {
                const auto rangev = (direction == Direction::Forward)
                                        ? range{i, nQ}
                                        : range{0, nQ - k + i + 1};
                const auto v = slice(V, rangev, i);
                auto Ci = rows(C, rangev);
                larf_work(LEFT_SIDE, direction, COLUMNWISE_STORAGE, v,
                          (trans == Op::ConjTrans) ? conj(tau[i]) : tau[i], Ci,
                          work);
            }
        }
        else {
            for (idx_t i = i0; i != iN; i += inc) {
                const auto rangev = (direction == Direction::Forward)
                                        ? range{i, nQ}
                                        : range{0, nQ - k + i + 1};
                const auto v = slice(V, rangev, i);
                auto Ci = cols(C, rangev);
                larf_work(RIGHT_SIDE, direction, COLUMNWISE_STORAGE, v,
                          (trans == Op::ConjTrans) ? conj(tau[i]) : tau[i], Ci,
                          work);
            }
        }
    }
    else {
        if (side == Side::Left) {
            for (idx_t i = i0; i != iN; i += inc) {
                const auto rangev = (direction == Direction::Forward)
                                        ? range{i, nQ}
                                        : range{0, nQ - k + i + 1};
                const auto v = slice(V, i, rangev);
                auto Ci = rows(C, rangev);
                larf_work(LEFT_SIDE, direction, ROWWISE_STORAGE, v,
                          (trans == Op::NoTrans) ? conj(tau[i]) : tau[i], Ci,
                          work);
            }
        }
        else {
            for (idx_t i = i0; i != iN; i += inc) {
                const auto rangev = (direction == Direction::Forward)
                                        ? range{i, nQ}
                                        : range{0, nQ - k + i + 1};
                const auto v = slice(V, i, rangev);
                auto Ci = cols(C, rangev);
                larf_work(RIGHT_SIDE, direction, ROWWISE_STORAGE, v,
                          (trans == Op::NoTrans) ? conj(tau[i]) : tau[i], Ci,
                          work);
            }
        }
    }

    return 0;
}

/**
 * @brief Applies unitary matrix Q to a matrix C.
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
 * @return 0 if success.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int unmq_level2(side_t side,
                trans_t trans,
                direction_t direction,
                storage_t storeMode,
                const matrixV_t& V,
                const vector_t& tau,
                matrixC_t& C)
{
    using idx_t = size_type<matrixC_t>;
    using work_t = matrix_type<matrixV_t, matrixC_t>;
    using T = type_t<work_t>;

    // Functor
    Create<work_t> new_matrix;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);

    // quick return
    if ((m <= 0) || (n <= 0) || (k <= 0)) return 0;

    // Allocates workspace
    WorkInfo workinfo =
        unmq_level2_worksize<T>(side, trans, direction, storeMode, V, tau, C);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return unmq_level2_work(side, trans, direction, storeMode, V, tau, C, work);
}

}  // namespace tlapack

#endif  // TLAPACK_UNMQ_LV2_HH
