/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARF_HH
#define TLAPACK_LARF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/ger.hpp"
#include "tlapack/blas/geru.hpp"

namespace tlapack {

template <TLAPACK_SIDE side_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_VECTOR vectorw_t,
          TLAPACK_SCALAR tau_t,
          TLAPACK_VECTOR vectorC0_t,
          TLAPACK_MATRIX matrixC1_t,
          enable_if_t<std::is_convertible_v<storage_t, StoreV>, int> = 0>
void larf(side_t side,
          storage_t storeMode,
          vector_t const& x,
          const tau_t& tau,
          vectorC0_t& C0,
          matrixC1_t& C1,
          vectorw_t& w)
{
    // data traits
    using idx_t = size_type<vectorC0_t>;
    using T = type_t<vectorw_t>;
    using real_t = real_type<T>;

    // constants
    const real_t one(1);
    const idx_t k = size(C0);
    const idx_t m = nrows(C1);
    const idx_t n = ncols(C1);

    // check arguments
    tlapack_check(side == Side::Left || side == Side::Right);
    tlapack_check(storeMode == StoreV::Columnwise ||
                  storeMode == StoreV::Rowwise);
    tlapack_check((idx_t)size(x) == (side == Side::Left) ? m : n);

    // Quick return if possible
    if (m == 0 || n == 0) {
        for (idx_t i = 0; i < k; ++i)
            C0[i] -= tau * C0[i];
        return;
    }

    if (side == Side::Left) {
        if (storeMode == StoreV::Columnwise) {
            // w := C0^H + C1^H*x
            for (idx_t i = 0; i < k; ++i)
                w[i] = conj(C0[i]);
            gemv(Op::ConjTrans, one, C1, x, one, w);

            // C1 := C1 - tau*x*w^H
            ger(-tau, x, w, C1);

            // C0 := C0 - tau*w^H
            for (idx_t i = 0; i < k; ++i)
                C0[i] -= tau * conj(w[i]);
        }
        else {
            // w := C0^t + C1^t*x
            for (idx_t i = 0; i < k; ++i)
                w[i] = C0[i];
            gemv(Op::Trans, one, C1, x, one, w);

            // C1 := C1 - tau*conj(x)*w^t
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    C1(i, j) -= tau * conj(x[i]) * w[j];

            // C0 := C0 - tau*w^t
            for (idx_t i = 0; i < k; ++i)
                C0[i] -= tau * w[i];
        }
    }
    else {  // side == Side::Right
        if (storeMode == StoreV::Columnwise) {
            // w := C0 + C1*x
            for (idx_t i = 0; i < k; ++i)
                w[i] = C0[i];
            gemv(Op::NoTrans, one, C1, x, one, w);

            // C1 := C1 - tau*w*x^H
            ger(-tau, w, x, C1);

            // C0 := C0 - tau*w
            for (idx_t i = 0; i < k; ++i)
                C0[i] -= tau * w[i];
        }
        else {
            // w := C0 + C1*conj(x)
            for (idx_t i = 0; i < k; ++i)
                w[i] = C0[i];
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    w[i] += C1(i, j) * conj(x[j]);

            // C1 := C1 - tau*w*x^t
            geru(-tau, w, x, C1);

            // C0 := C0 - tau*w
            for (idx_t i = 0; i < k; ++i)
                C0[i] -= tau * w[i];
        }
    }
}

/** Worspace query of larf().
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ from the Left.
 *     - Side::Right: apply $H$ from the Right.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] x Vector of size m-1 if side = Side::Left,
 *                          or n-1 if side = Side::Right.
 *
 * @param[in] tau Value of tau in the representation of H.
 *
 * @param[in] C0
 *     On entry, the m-by-n matrix C.
 *
 * @param[in] C1
 *     On entry, the m-by-n matrix C.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SCALAR tau_t,
          TLAPACK_VECTOR vectorC0_t,
          TLAPACK_MATRIX matrixC1_t,
          enable_if_t<std::is_convertible_v<storage_t, StoreV>, int> = 0>
inline constexpr WorkInfo larf_worksize(side_t side,
                                        storage_t storeMode,
                                        vector_t const& x,
                                        const tau_t& tau,
                                        const vectorC0_t& C0,
                                        const matrixC1_t& C1,
                                        const WorkspaceOpts& opts = {})
{
    using work_t = vector_type<vectorC0_t, matrixC1_t, vector_t>;
    using idx_t = size_type<vectorC0_t>;
    using T = type_t<work_t>;

    // constants
    const idx_t k = size(C0);

    return WorkInfo(sizeof(T), k);
}

/** Applies an elementary reflector defined by tau and v to a m-by-n matrix C
 * decomposed into C0 and C1.
 * \[
 *      C_0 = (1-\tau) C_0 - \tau v^H C_1, \\
 *      C_1 = -\tau v C_0 + (I-\tau vv^H) C_1,
 * \]
 * if side = Side::Left, or
 * \[
 *      C_0 = (1-\tau) C_0 -\tau C_1 v, \\
 *      C_1 = -\tau C_0 v^H + C_1 (I-\tau vv^H),
 * \]
 * if side = Side::Right.
 *
 * The elementary reflector is defined as
 * \[
 * H =
 * \begin{bmatrix}
 *      1-\tau & -\tau v^H \\
 *      -\tau v & I-\tau vv^H
 * \end{bmatrix}
 * \]
 *
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ from the Left.
 *     - Side::Right: apply $H$ from the Right.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] x Vector $v$ if storeMode = StoreV::Columnwise, or
 *                     $v^H$ if storeMode = StoreV::Rowwise.
 *
 * @param[in] tau Value of tau in the representation of H.
 *
 * @param[in,out] C0 Vector of size m-1 if side = Side::Left,
 *                               or n-1 if side = Side::Right.
 *     On exit, C0 is overwritten by
 *      - $(1-\tau) C_0 - \tau v^H C_1$ if side = Side::Left, or
 *      - $(1-\tau) C_0 -\tau C_1 v$ if side = Side::Right.
 *
 * @param[in,out] C1 Matrix of size (m-1)-by-n if side = Side::Left,
 *                               or m-by-(n-1) if side = Side::Right.
 *     On exit, C1 is overwritten by
 *     - $-\tau v C_0 + (I-\tau vv^H) C_1$ if side = Side::Left, or
 *     - $-\tau C_0 v^H + C_1 (I-\tau vv^H)$ if side = Side::Right.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SCALAR tau_t,
          TLAPACK_VECTOR vectorC0_t,
          TLAPACK_MATRIX matrixC1_t,
          enable_if_t<std::is_convertible_v<storage_t, StoreV>, int> = 0>
void larf(side_t side,
          storage_t storeMode,
          vector_t const& x,
          const tau_t& tau,
          vectorC0_t& C0,
          matrixC1_t& C1,
          const WorkspaceOpts& opts = {})
{
    // data traits
    using work_t = vector_type<vectorC0_t, matrixC1_t, vector_t>;
    using idx_t = size_type<vectorC0_t>;

    // Functor
    Create<work_t> new_vector;

    // constants
    const idx_t k = size(C0);
    const idx_t m = nrows(C1);
    const idx_t n = ncols(C1);

    // check arguments
    tlapack_check(side == Side::Left || side == Side::Right);
    tlapack_check(storeMode == StoreV::Columnwise ||
                  storeMode == StoreV::Rowwise);
    tlapack_check((idx_t)size(x) == (side == Side::Left) ? m : n);

    // Quick return if possible
    if (m == 0 || n == 0) {
        for (idx_t i = 0; i < k; ++i)
            C0[i] -= tau * C0[i];
        return;
    }

    // Allocates workspace
    VectorOfBytes localworkdata;
    const Workspace work = [&]() {
        WorkInfo workinfo =
            larf_worksize(side, storeMode, x, tau, C0, C1, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();
    auto w = new_vector(work, k);

    return larf(side, storeMode, x, tau, C0, C1, w);
}

template <TLAPACK_SIDE side_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_VECTOR vectorw_t,
          TLAPACK_SCALAR tau_t,
          TLAPACK_SMATRIX matrix_t,
          enable_if_t<std::is_convertible_v<direction_t, Direction>, int> = 0>
inline void larf(side_t side,
                 direction_t direction,
                 storage_t storeMode,
                 vector_t const& v,
                 const tau_t& tau,
                 matrix_t& C,
                 vectorw_t& w)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    // check arguments
    tlapack_check(side == Side::Left || side == Side::Right);
    tlapack_check(direction == Direction::Backward ||
                  direction == Direction::Forward);
    tlapack_check(storeMode == StoreV::Columnwise ||
                  storeMode == StoreV::Rowwise);
    tlapack_check((idx_t)size(v) == ((side == Side::Left) ? m : n));

    // quick return
    if (m == 0 || n == 0) return;

    // The following code was changed from:
    //
    // if( side == Side::Left ) {
    //     gemv(Op::ConjTrans, one, C, v, work);
    //     ger(-tau, v, work, C);
    // }
    // else{
    //     gemv(Op::NoTrans, one, C, v, work);
    //     ger(-tau, work, v, C);
    // }
    //
    // This is so that v[0] doesn't need to be changed to 1,
    // which is better for thread safety.

    if (side == Side::Left) {
        auto C0 = (direction == Direction::Forward) ? row(C, 0) : row(C, m - 1);
        auto C1 = (direction == Direction::Forward) ? rows(C, range{1, m})
                                                    : rows(C, range{0, m - 1});
        auto x = (direction == Direction::Forward) ? slice(v, range{1, m})
                                                   : slice(v, range{0, m - 1});
        larf(side, storeMode, x, tau, C0, C1, w);
    }
    else {  // side == Side::Right
        auto C0 = (direction == Direction::Forward) ? col(C, 0) : col(C, n - 1);
        auto C1 = (direction == Direction::Forward) ? cols(C, range{1, n})
                                                    : cols(C, range{0, n - 1});
        auto x = (direction == Direction::Forward) ? slice(v, range{1, n})
                                                   : slice(v, range{0, n - 1});
        larf(side, storeMode, x, tau, C0, C1, w);
    }
}

/** Worspace query of larf().
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ from the Left.
 *     - Side::Right: apply $H$ from the Right.
 *
 * @param[in] direction
 *     v = [ 1 x ] if direction == Direction::Forward and
 *     v = [ x 1 ] if direction == Direction::Backward.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] v Vector of size m if side = Side::Left,
 *                          or n if side = Side::Right.
 *
 * @param[in] tau Value of tau in the representation of H.
 *
 * @param[in] C
 *     On entry, the m-by-n matrix C.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SCALAR tau_t,
          TLAPACK_SMATRIX matrix_t,
          enable_if_t<std::is_convertible_v<direction_t, Direction>, int> = 0>
inline constexpr WorkInfo larf_worksize(side_t side,
                                        direction_t direction,
                                        storage_t storeMode,
                                        vector_t const& v,
                                        const tau_t& tau,
                                        const matrix_t& C,
                                        const WorkspaceOpts& opts = {})
{
    using work_t = vector_type<matrix_t, vector_t>;
    using idx_t = size_type<matrix_t>;
    using T = type_t<work_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    return WorkInfo(sizeof(T), (side == Side::Left) ? n : m);
}

/** Applies an elementary reflector H to a m-by-n matrix C.
 *
 * The elementary reflector H can be applied on either the left or right, with
 * \[
 *        H = I - \tau v v^H.
 * \]
 * where v = [ 1 x ] if direction == Direction::Forward and
 *       v = [ x 1 ] if direction == Direction::Backward.
 *
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ from the Left.
 *     - Side::Right: apply $H$ from the Right.
 *
 * @param[in] direction
 *     v = [ 1 x ] if direction == Direction::Forward and
 *     v = [ x 1 ] if direction == Direction::Backward.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] v Vector of size m if side = Side::Left,
 *                          or n if side = Side::Right.
 *
 * @param[in] tau Value of tau in the representation of H.
 *
 * @param[in,out] C
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by $H C$ if side = Side::Left,
 *                               or $C H$ if side = Side::Right.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SCALAR tau_t,
          TLAPACK_SMATRIX matrix_t,
          enable_if_t<std::is_convertible_v<direction_t, Direction>, int> = 0>
inline void larf(side_t side,
                 direction_t direction,
                 storage_t storeMode,
                 vector_t const& v,
                 const tau_t& tau,
                 matrix_t& C,
                 const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    // check arguments
    tlapack_check(side == Side::Left || side == Side::Right);
    tlapack_check(direction == Direction::Backward ||
                  direction == Direction::Forward);
    tlapack_check(storeMode == StoreV::Columnwise ||
                  storeMode == StoreV::Rowwise);
    tlapack_check((idx_t)size(v) == ((side == Side::Left) ? m : n));

    // quick return
    if (m == 0 || n == 0) return;

    // The following code was changed from:
    //
    // if( side == Side::Left ) {
    //     gemv(Op::ConjTrans, one, C, v, work);
    //     ger(-tau, v, work, C);
    // }
    // else{
    //     gemv(Op::NoTrans, one, C, v, work);
    //     ger(-tau, work, v, C);
    // }
    //
    // This is so that v[0] doesn't need to be changed to 1,
    // which is better for thread safety.

    if (side == Side::Left) {
        auto C0 = (direction == Direction::Forward) ? row(C, 0) : row(C, m - 1);
        auto C1 = (direction == Direction::Forward) ? rows(C, range{1, m})
                                                    : rows(C, range{0, m - 1});
        auto x = (direction == Direction::Forward) ? slice(v, range{1, m})
                                                   : slice(v, range{0, m - 1});
        larf(side, storeMode, x, tau, C0, C1, opts);
    }
    else {  // side == Side::Right
        auto C0 = (direction == Direction::Forward) ? col(C, 0) : col(C, n - 1);
        auto C1 = (direction == Direction::Forward) ? cols(C, range{1, n})
                                                    : cols(C, range{0, n - 1});
        auto x = (direction == Direction::Forward) ? slice(v, range{1, n})
                                                   : slice(v, range{0, n - 1});
        larf(side, storeMode, x, tau, C0, C1, opts);
    }
}

}  // namespace tlapack

#endif  // TLAPACK_LARF_HH
