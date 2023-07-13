/// @file unml2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/orml2.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNML2_HH
#define TLAPACK_UNML2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Worspace query of unml2()
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
 * @param[in] A
 *      - side = Side::Left:    k-by-m matrix;
 *      - side = Side::Right:   k-by-n matrix.
 *
 * @param[in] tau Vector of length k
 *      Contains the scalar factors of the elementary reflectors.
 *
 * @param[in] C m-by-n matrix.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SMATRIX matrixA_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_VECTOR tau_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t>
inline constexpr WorkInfo unml2_worksize(side_t side,
                                         trans_t trans,
                                         const matrixA_t& A,
                                         const tau_t& tau,
                                         const matrixC_t& C,
                                         const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t nA = (side == Side::Left) ? m : n;

    auto v = slice(A, 0, range{0, nA});
    return larf_worksize(side, FORWARD, ROWWISE_STORAGE, v, tau[0], C, opts);
}

/** Applies unitary matrix Q from an LQ factorization to a matrix C.
 *
 * - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 * - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 * - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 * - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * as returned by gelqf
 *
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 * @tparam trans_t Either Op or any class that implements `operator Op()`.
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
 * @param[in] A
 *      - side = Side::Left:    k-by-m matrix;
 *      - side = Side::Right:   k-by-n matrix.
 *
 * @param[in] tau Vector of length k
 *      Contains the scalar factors of the elementary reflectors.
 *
 * @param[in,out] C m-by-n matrix.
 *      On exit, C is replaced by one of the following:
 *      - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 *      - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 *      - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 *      - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrixA_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_VECTOR tau_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t>
int unml2(side_t side,
          trans_t trans,
          const matrixA_t& A,
          const tau_t& tau,
          matrixC_t& C,
          const WorkspaceOpts& opts = {})
{
    using TA = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nA = (side == Side::Left) ? m : n;

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans);
    tlapack_check_false(trans == Op::Trans && is_complex<TA>);

    // quick return
    if ((m == 0) || (n == 0) || (k == 0)) return 0;

    // Allocates workspace
    VectorOfBytes localworkdata;
    Workspace work = [&]() {
        WorkInfo workinfo = unml2_worksize(side, trans, A, tau, C, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = WorkspaceOpts{work};

    // const expressions
    const bool positiveInc =
        (((side == Side::Left) && (trans == Op::NoTrans)) ||
         (!(side == Side::Left) && !(trans == Op::NoTrans)));
    const idx_t i0 = (positiveInc) ? 0 : k - 1;
    const idx_t iN = (positiveInc) ? k : -1;
    const idx_t inc = (positiveInc) ? 1 : -1;

    // Main loop
    for (idx_t i = i0; i != iN; i += inc) {
        auto v = slice(A, i, range{i, nA});

        if (side == Side::Left) {
            auto Ci = rows(C, range{i, m});
            larf(LEFT_SIDE, FORWARD, ROWWISE_STORAGE, v,
                 (trans == Op::NoTrans) ? conj(tau[i]) : tau[i], Ci, larfOpts);
        }
        else {
            auto Ci = cols(C, range{i, n});
            larf(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, v,
                 (trans == Op::NoTrans) ? conj(tau[i]) : tau[i], Ci, larfOpts);
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNML2_HH
