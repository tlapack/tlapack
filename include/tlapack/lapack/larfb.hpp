/// @file larfb.hpp Applies a Householder block reflector to a matrix.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larfb.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARFB_HH
#define TLAPACK_LARFB_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/lacpy.hpp"

namespace tlapack {

/** Worspace query of larfb()
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ or $H^H$ from the Left.
 *     - Side::Right: apply $H$ or $H^H$ from the Right.
 *
 * @param[in] trans
 *     - Op::NoTrans:   apply $H  $ (No transpose).
 *     - Op::Trans:     apply $H^T$ (Transpose, only allowed if the type of H is
 * Real).
 *     - Op::ConjTrans: apply $H^H$ (Conjugate transpose).
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
 *     See Further Details.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise:
 *       - if side = Side::Left,  the m-by-k matrix V;
 *       - if side = Side::Right, the n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:
 *       - if side = Side::Left,  the k-by-m matrix V;
 *       - if side = Side::Right, the k-by-n matrix V.
 *
 * @param[in] Tmatrix
 *     The k-by-k matrix T.
 *     The triangular k-by-k matrix T in the representation of the block
 * reflector.
 *
 * @param[in] C
 *     On entry, the m-by-n matrix C.
 *
 * @param[in] opts Options.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_MATRIX matrixV_t,
          TLAPACK_MATRIX matrixT_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          class direction_t,
          TLAPACK_STOREV storage_t,
          class workW_t = void>
inline constexpr void larfb_worksize(side_t side,
                                     trans_t trans,
                                     direction_t direction,
                                     storage_t storeMode,
                                     const matrixV_t& V,
                                     const matrixT_t& Tmatrix,
                                     const matrixC_t& C,
                                     workinfo_t& workinfo,
                                     const workspace_opts_t<workW_t>& opts = {})
{
    using idx_t = size_type<matrixC_t>;
    using matrixW_t =
        deduce_work_t<workW_t, matrix_type<matrixV_t, matrixC_t> >;
    using T = type_t<matrixW_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = nrows(Tmatrix);

    workinfo_t myWorkinfo;
    if (layout<matrixW_t> == Layout::RowMajor) {
        myWorkinfo.m = (side == Side::Left) ? k : m;
        myWorkinfo.n = sizeof(T) * ((side == Side::Left) ? n : k);
    }
    else if (layout<matrixW_t> == Layout::ColMajor) {
        myWorkinfo.m = sizeof(T) * ((side == Side::Left) ? k : m);
        myWorkinfo.n = (side == Side::Left) ? n : k;
    }
    else {
        myWorkinfo.m = sizeof(T);
        myWorkinfo.n = (side == Side::Left) ? k * n : m * k;
    }

    workinfo.minMax(myWorkinfo);
}

/** Applies a block reflector $H$ or its conjugate transpose $H^H$ to a
 * m-by-n matrix C, from either the left or the right.
 *
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 * @tparam trans_t Either Op or any class that implements `operator Op()`.
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ or $H^H$ from the Left.
 *     - Side::Right: apply $H$ or $H^H$ from the Right.
 *
 * @param[in] trans
 *     - Op::NoTrans:   apply $H  $ (No transpose).
 *     - Op::Trans:     apply $H^T$ (Transpose, only allowed if the type of H is
 * Real).
 *     - Op::ConjTrans: apply $H^H$ (Conjugate transpose).
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
 *     See Further Details.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise:
 *       - if side = Side::Left,  the m-by-k matrix V;
 *       - if side = Side::Right, the n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:
 *       - if side = Side::Left,  the k-by-m matrix V;
 *       - if side = Side::Right, the k-by-n matrix V.
 *
 * @param[in] Tmatrix
 *     The k-by-k matrix T.
 *     The triangular k-by-k matrix T in the representation of the block
 * reflector.
 *
 * @param[in,out] C
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by $H C$ or $H^H C$ or $C H$ or $C H^H$.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @par Further Details
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
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrixV_t,
          TLAPACK_MATRIX matrixT_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          class direction_t,
          TLAPACK_STOREV storage_t,
          class workW_t = void>
int larfb(side_t side,
          trans_t trans,
          direction_t direction,
          storage_t storeMode,
          const matrixV_t& V,
          const matrixT_t& Tmatrix,
          matrixC_t& C,
          const workspace_opts_t<workW_t>& opts = {})
{
    using idx_t = size_type<matrixC_t>;
    using matrixW_t =
        deduce_work_t<workW_t, matrix_type<matrixV_t, matrixC_t> >;
    using T = type_t<matrixW_t>;
    using real_t = real_type<T>;

    using pair = pair<idx_t, idx_t>;

    // Functor
    Create<matrixW_t> new_matrix;

    // constants
    const real_t one(1);
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = nrows(Tmatrix);

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(
        trans != Op::NoTrans && trans != Op::ConjTrans &&
        ((trans != Op::Trans) || is_complex<type_t<matrixV_t> >::value));
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check_false(storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise);
    tlapack_check(
        (storeMode == StoreV::Columnwise)
            ? ((ncols(V) == k) && (side == Side::Left) ? (nrows(V) == m)
                                                       : (nrows(V) == n))
            : ((nrows(V) == k) && (side == Side::Left) ? (ncols(V) == m)
                                                       : (ncols(V) == n)));
    tlapack_check(nrows(Tmatrix) == ncols(Tmatrix));

    // Quick return
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    const Workspace work = [&]() {
        workinfo_t workinfo;
        larfb_worksize(side, trans, direction, storeMode, V, Tmatrix, C,
                       workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    if (storeMode == StoreV::Columnwise) {
        if (direction == Direction::Forward) {
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                const auto V1 = rows(V, pair{0, k});
                const auto V2 = rows(V, pair{k, m});
                auto C1 = rows(C, pair{0, k});
                auto C2 = rows(C, pair{k, m});
                auto W = new_matrix(work, k, n);

                // W := C1
                lacpy(dense, C1, W);
                // W := V1^H W
                trmm(side, Uplo::Lower, Op::ConjTrans, Diag::Unit, one, V1, W);
                if (m > k)
                    // W := W + V2^H C2
                    gemm(Op::ConjTrans, Op::NoTrans, one, V2, C2, one, W);
                // W := op(Tmatrix) W
                trmm(side, Uplo::Upper, trans, Diag::NonUnit, one, Tmatrix, W);
                if (m > k)
                    // C2 := C2 - V2 W
                    gemm(Op::NoTrans, Op::NoTrans, -one, V2, W, one, C2);
                // W := - V1 W
                trmm(side, Uplo::Lower, Op::NoTrans, Diag::Unit, -one, V1, W);

                // C1 := C1 + W
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < k; ++i)
                        C1(i, j) += W(i, j);
            }
            else {  // side == Side::Right
                // W is an m-by-k matrix
                // V is an n-by-k matrix

                // Matrix views
                const auto V1 = rows(V, pair{0, k});
                const auto V2 = rows(V, pair{k, n});
                auto C1 = cols(C, pair{0, k});
                auto C2 = cols(C, pair{k, n});
                auto W = new_matrix(work, m, k);

                // W := C1
                lacpy(dense, C1, W);
                // W := W V1
                trmm(side, Uplo::Lower, Op::NoTrans, Diag::Unit, one, V1, W);
                if (n > k)
                    // W := W + C2 V2
                    gemm(Op::NoTrans, Op::NoTrans, one, C2, V2, one, W);
                // W := W op(Tmatrix)
                trmm(Side::Right, Uplo::Upper, trans, Diag::NonUnit, one,
                     Tmatrix, W);
                if (n > k)
                    // C2 := C2 - W V2^H
                    gemm(Op::NoTrans, Op::ConjTrans, -one, W, V2, one, C2);
                // W := - W V1^H
                trmm(side, Uplo::Lower, Op::ConjTrans, Diag::Unit, -one, V1, W);

                // C1 := C1 + W
                for (idx_t j = 0; j < k; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        C1(i, j) += W(i, j);
            }
        }
        else {  // direct == Direction::Backward
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                const auto V1 = rows(V, pair{0, m - k});
                const auto V2 = rows(V, pair{m - k, m});
                auto C1 = rows(C, pair{0, m - k});
                auto C2 = rows(C, pair{m - k, m});
                auto W = new_matrix(work, k, n);

                // W := C2
                lacpy(dense, C2, W);
                // W := V2^H W
                trmm(side, Uplo::Upper, Op::ConjTrans, Diag::Unit, one, V2, W);
                if (m > k)
                    // W := W + V1^H C1
                    gemm(Op::ConjTrans, Op::NoTrans, one, V1, C1, one, W);
                // W := op(Tmatrix) W
                trmm(side, Uplo::Lower, trans, Diag::NonUnit, one, Tmatrix, W);
                if (m > k)
                    // C1 := C1 - V1 W
                    gemm(Op::NoTrans, Op::NoTrans, -one, V1, W, one, C1);
                // W := - V2 W
                trmm(side, Uplo::Upper, Op::NoTrans, Diag::Unit, -one, V2, W);

                // C2 := C2 + W
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < k; ++i)
                        C2(i, j) += W(i, j);
            }
            else {  // side == Side::Right
                // W is an m-by-k matrix
                // V is an n-by-k matrix

                // Matrix views
                const auto V1 = rows(V, pair{0, n - k});
                const auto V2 = rows(V, pair{n - k, n});
                auto C1 = cols(C, pair{0, n - k});
                auto C2 = cols(C, pair{n - k, n});
                auto W = new_matrix(work, m, k);

                // W := C2
                lacpy(dense, C2, W);
                // W := W V2
                trmm(side, Uplo::Upper, Op::NoTrans, Diag::Unit, one, V2, W);
                if (n > k)
                    // W := W + C1 V1
                    gemm(Op::NoTrans, Op::NoTrans, one, C1, V1, one, W);
                // W := W op(Tmatrix)
                trmm(side, Uplo::Lower, trans, Diag::NonUnit, one, Tmatrix, W);
                if (n > k)
                    // C1 := C1 - W V1^H
                    gemm(Op::NoTrans, Op::ConjTrans, -one, W, V1, one, C1);
                // W := - W V2^H
                trmm(side, Uplo::Upper, Op::ConjTrans, Diag::Unit, -one, V2, W);

                // C2 := C2 + W
                for (idx_t j = 0; j < k; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        C2(i, j) += W(i, j);
            }
        }
    }
    else {  // storeV == StoreV::Rowwise
        if (direction == Direction::Forward) {
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                const auto V1 = cols(V, pair{0, k});
                const auto V2 = cols(V, pair{k, m});
                auto C1 = rows(C, pair{0, k});
                auto C2 = rows(C, pair{k, m});
                auto W = new_matrix(work, k, n);

                // W := C1
                lacpy(dense, C1, W);
                // W := V1 W
                trmm(side, Uplo::Upper, Op::NoTrans, Diag::Unit, one, V1, W);
                if (m > k)
                    // W := W + V2 C2
                    gemm(Op::NoTrans, Op::NoTrans, one, V2, C2, one, W);
                // W := op(Tmatrix) W
                trmm(side, Uplo::Upper, trans, Diag::NonUnit, one, Tmatrix, W);
                if (m > k)
                    // C2 := C2 - V2^H W
                    gemm(Op::ConjTrans, Op::NoTrans, -one, V2, W, one, C2);
                // W := - V1^H W
                trmm(side, Uplo::Upper, Op::ConjTrans, Diag::Unit, -one, V1, W);

                // C1 := C1 - W
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < k; ++i)
                        C1(i, j) += W(i, j);
            }
            else {  // side == Side::Right
                // W is an m-by-k matrix
                // V is an k-by-n matrix

                // Matrix views
                const auto V1 = cols(V, pair{0, k});
                const auto V2 = cols(V, pair{k, n});
                auto C1 = cols(C, pair{0, k});
                auto C2 = cols(C, pair{k, n});
                auto W = new_matrix(work, m, k);

                // W := C1
                lacpy(dense, C1, W);
                // W := W V1^H
                trmm(side, Uplo::Upper, Op::ConjTrans, Diag::Unit, one, V1, W);
                if (n > k)
                    // W := W + C2 V2^H
                    gemm(Op::NoTrans, Op::ConjTrans, one, C2, V2, one, W);
                // W := W op(Tmatrix)
                trmm(side, Uplo::Upper, trans, Diag::NonUnit, one, Tmatrix, W);
                if (n > k)
                    // C2 := C2 - W V2
                    gemm(Op::NoTrans, Op::NoTrans, -one, W, V2, one, C2);
                // W := - W V1
                trmm(side, Uplo::Upper, Op::NoTrans, Diag::Unit, -one, V1, W);

                // C1 := C1 + W
                for (idx_t j = 0; j < k; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        C1(i, j) += W(i, j);
            }
        }
        else {  // direct == Direction::Backward
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                const auto V1 = cols(V, pair{0, m - k});
                const auto V2 = cols(V, pair{m - k, m});
                auto C1 = rows(C, pair{0, m - k});
                auto C2 = rows(C, pair{m - k, m});
                auto W = new_matrix(work, k, n);

                // W := C2
                lacpy(dense, C2, W);
                // W := V2 W
                trmm(side, Uplo::Lower, Op::NoTrans, Diag::Unit, one, V2, W);
                if (m > k)
                    // W := W + V1 C1
                    gemm(Op::NoTrans, Op::NoTrans, one, V1, C1, one, W);
                // W := op(Tmatrix) W
                trmm(side, Uplo::Lower, trans, Diag::NonUnit, one, Tmatrix, W);
                if (m > k)
                    // C1 := C1 - V1^H W
                    gemm(Op::ConjTrans, Op::NoTrans, -one, V1, W, one, C1);
                // W := - V2^H W
                trmm(side, Uplo::Lower, Op::ConjTrans, Diag::Unit, -one, V2, W);

                // C2 := C2 + W
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < k; ++i)
                        C2(i, j) += W(i, j);
            }
            else {  // side == Side::Right
                // W is an m-by-k matrix
                // V is an k-by-n matrix

                // Matrix views
                const auto V1 = cols(V, pair{0, n - k});
                const auto V2 = cols(V, pair{n - k, n});
                auto C1 = cols(C, pair{0, n - k});
                auto C2 = cols(C, pair{n - k, n});
                auto W = new_matrix(work, m, k);

                // W := C2
                lacpy(dense, C2, W);
                // W := W V2^H
                trmm(side, Uplo::Lower, Op::ConjTrans, Diag::Unit, one, V2, W);
                if (n > k)
                    // W := W + C1 V1^H
                    gemm(Op::NoTrans, Op::ConjTrans, one, C1, V1, one, W);
                // W := W op(Tmatrix)
                trmm(side, Uplo::Lower, trans, Diag::NonUnit, one, Tmatrix, W);
                if (n > k)
                    // C1 := C1 - W V1
                    gemm(Op::NoTrans, Op::NoTrans, -one, W, V1, one, C1);
                // W := - W V2
                trmm(side, Uplo::Lower, Op::NoTrans, Diag::Unit, -one, V2, W);

                // C2 := C2 + W
                for (idx_t j = 0; j < k; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        C2(i, j) += W(i, j);
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LARFB_HH
