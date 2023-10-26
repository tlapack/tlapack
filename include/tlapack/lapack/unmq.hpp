/// @file unmq.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zunmqr.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNMQ_HH
#define TLAPACK_UNMQ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"

namespace tlapack {

/**
 * Options struct for unmq
 */
struct UnmqOpts {
    size_t nb = 32;  ///< Block size
};

/** Worspace query of unmq()
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
constexpr WorkInfo unmq_worksize(side_t side,
                                 trans_t trans,
                                 direction_t direction,
                                 storage_t storeMode,
                                 const matrixV_t& V,
                                 const vector_t& tau,
                                 const matrixC_t& C,
                                 const UnmqOpts& opts = {})
{
    using idx_t = size_type<matrixC_t>;
    using work_t = matrix_type<matrixV_t, vector_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nQ = (side == Side::Left) ? m : n;
    const idx_t nb = min((idx_t)opts.nb, k);

    // Local workspace sizes
    WorkInfo workinfo =
        (is_same_v<T, type_t<work_t>>) ? WorkInfo(nb, nb) : WorkInfo(0);

    auto&& Vi = (storeMode == StoreV::Columnwise)
                    ? slice(V, range{0, nQ}, range{0, nb})
                    : slice(V, range{0, nb}, range{0, nQ});
    auto&& matrixTi = slice(V, range{0, nb}, range{0, nb});

    // larfb:
    workinfo += larfb_worksize<T>(side, NO_TRANS, direction, storeMode, Vi,
                                  matrixTi, C);

    return workinfo;
}

/** @copybrief unmq()
 * Workspace is provided as an argument.
 * @copydetails unmq()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_WORKSPACE work_t>
int unmq_work(side_t side,
              trans_t trans,
              direction_t direction,
              storage_t storeMode,
              const matrixV_t& V,
              const vector_t& tau,
              matrixC_t& C,
              work_t& work,
              const UnmqOpts& opts = {})
{
    using idx_t = size_type<matrixC_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nQ = (side == Side::Left) ? m : n;
    const idx_t nb = min((idx_t)opts.nb, k);

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

    // Matrix matrixT
    auto [matrixT, work1] = reshape(work, nb, nb);

    // const expressions
    const bool positiveIncLeft =
        (storeMode == StoreV::Columnwise)
            ? ((direction == Direction::Backward) ? (trans == Op::NoTrans)
                                                  : (trans != Op::NoTrans))
            : ((direction == Direction::Forward) ? (trans == Op::NoTrans)
                                                 : (trans != Op::NoTrans));
    const bool positiveInc =
        (side == Side::Left) ? positiveIncLeft : !positiveIncLeft;
    const idx_t i0 = (positiveInc) ? 0 : ((k - 1) / nb) * nb;
    const idx_t iN = (positiveInc) ? ((k - 1) / nb + 1) * nb : -nb;
    const idx_t inc = (positiveInc) ? nb : -nb;

    if (storeMode == StoreV::Columnwise) {
        for (idx_t i = i0; i != iN; i += inc) {
            const idx_t ib = min(nb, k - i);
            const auto rangev = (direction == Direction::Forward)
                                    ? range{i, nQ}
                                    : range{0, nQ - k + i + ib};
            const auto Vi = slice(V, rangev, range{i, i + ib});
            const auto taui = slice(tau, range{i, i + ib});
            auto matrixTi = slice(matrixT, range{0, ib}, range{0, ib});

            // Form the triangular factor of the block reflector
            larft(direction, COLUMNWISE_STORAGE, Vi, taui, matrixTi);

            // H or H**H is applied to either C[i:m,0:n] or C[0:m,i:n]
            auto Ci = (side == Side::Left) ? slice(C, rangev, range{0, n})
                                           : slice(C, range{0, m}, rangev);
            larfb_work(side, trans, direction, COLUMNWISE_STORAGE, Vi, matrixTi,
                       Ci, work1);
        }
    }
    else {
        for (idx_t i = i0; i != iN; i += inc) {
            const idx_t ib = min(nb, k - i);
            const auto rangev = (direction == Direction::Forward)
                                    ? range{i, nQ}
                                    : range{0, nQ - k + i + ib};
            const auto Vi = slice(V, range{i, i + ib}, rangev);
            const auto taui = slice(tau, range{i, i + ib});
            auto matrixTi = slice(matrixT, range{0, ib}, range{0, ib});

            // Form the triangular factor of the block reflector
            larft(direction, ROWWISE_STORAGE, Vi, taui, matrixTi);

            // H or H**H is applied to either C[i:m,0:n] or C[0:m,i:n]
            auto Ci = (side == Side::Left) ? slice(C, rangev, range{0, n})
                                           : slice(C, range{0, m}, rangev);
            larfb_work(side,
                       (trans == Op::NoTrans) ? Op::ConjTrans : Op::NoTrans,
                       direction, ROWWISE_STORAGE, Vi, matrixTi, Ci, work1);
        }
    }

    return 0;
}

/**
 * @brief Applies unitary matrix Q to a matrix C. Blocked algorithm.
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
 *
 * @return 0 if success.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int unmq(side_t side,
         trans_t trans,
         direction_t direction,
         storage_t storeMode,
         const matrixV_t& V,
         const vector_t& tau,
         matrixC_t& C,
         const UnmqOpts& opts = {})
{
    using idx_t = size_type<matrixC_t>;
    using work_t = matrix_type<matrixV_t, vector_t>;
    using T = type_t<work_t>;

    // Functor
    Create<work_t> new_matrix;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);

    // quick return
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo =
        unmq_worksize<T>(side, trans, direction, storeMode, V, tau, C, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return unmq_work(side, trans, direction, storeMode, V, tau, C, work, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_UNMQ_HH
