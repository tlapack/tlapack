/// @file unmql.hpp Multiplies the general m-by-n matrix C by Q from
/// tlapack::geqlf()
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNMQL_HH
#define TLAPACK_UNMQL_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"

namespace tlapack {

/**
 * Options struct for unmql
 */
template <class workT_t = void>
struct unmql_opts_t : public workspace_opts_t<workT_t> {
    inline constexpr unmql_opts_t(const workspace_opts_t<workT_t>& opts = {})
        : workspace_opts_t<workT_t>(opts){};

    size_type<workT_t> nb = 32;  ///< Block size
};

/** Applies unitary matrix Q from an QL factorization to a matrix C.
 *
 * - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 * - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 * - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 * - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * as returned by geqlf
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
 *      - side = Side::Left:    m-by-k matrix;
 *      - side = Side::Right:   n-by-k matrix.
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
 * @return workinfo_t The amount workspace required.
 *
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrixA_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR tau_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          class workT_t = void>
inline constexpr workinfo_t unmql_worksize(
    side_t side,
    trans_t trans,
    const matrixA_t& A,
    const tau_t& tau,
    const matrixC_t& C,
    const unmql_opts_t<workT_t>& opts = {})
{
    using idx_t = size_type<matrixC_t>;
    using matrixT_t = deduce_work_t<workT_t, matrix_type<matrixA_t, tau_t> >;
    using T = type_t<matrixT_t>;
    using pair = std::pair<idx_t, idx_t>;

    // Constants
    const idx_t k = size(tau);
    const idx_t nb = min<idx_t>(opts.nb, k);

    // Local workspace sizes
    workinfo_t workinfo(nb * sizeof(T), nb);

    // larfb:
    {
        // Constants
        const idx_t m = nrows(C);
        const idx_t n = ncols(C);
        const idx_t nA = (side == Side::Left) ? m : n;

        // Empty matrices
        const auto V = slice(A, pair{0, nA}, pair{0, nb});
        const auto matrixT = slice(A, pair{0, nb}, pair{0, nb});

        // Internal workspace queries
        workinfo += larfb_worksize(side, trans, backward, columnwise_storage, V,
                                   matrixT, C, opts);
    }

    return workinfo;
}

/** Applies orthogonal matrix op(Q) to a matrix C using a blocked code.
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
 *      - side = Side::Left:    m-by-k matrix;
 *      - side = Side::Right:   n-by-k matrix.
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
          TLAPACK_SVECTOR tau_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          class workT_t = void>
int unmql(side_t side,
          trans_t trans,
          const matrixA_t& A,
          const tau_t& tau,
          matrixC_t& C,
          const unmql_opts_t<workT_t>& opts = {})
{
    using TA = type_t<matrixA_t>;
    using idx_t = size_type<matrixC_t>;
    using matrixT_t = deduce_work_t<workT_t, matrix_type<matrixA_t, tau_t> >;

    using pair = pair<idx_t, idx_t>;

    // Functor
    Create<matrixT_t> new_matrix;

    // Constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nA = (side == Side::Left) ? m : n;
    const idx_t nb = min<idx_t>(opts.nb, k);

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans);
    tlapack_check_false(trans == Op::Trans && is_complex<TA>::value);

    // quick return
    if ((m == 0) || (n == 0) || (k == 0)) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo = unmql_worksize(side, trans, A, tau, C, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Preparing loop indexes
    const bool positiveInc =
        (((side == Side::Left) && (trans == Op::NoTrans)) ||
         (!(side == Side::Left) && !(trans == Op::NoTrans)));
    const idx_t i0 = (positiveInc) ? 0 : ((k - 1) / nb) * nb;
    const idx_t iN = (positiveInc) ? ((k - 1) / nb + 1) * nb : -nb;
    const idx_t inc = (positiveInc) ? nb : -nb;

    // Matrix T and recompute work
    Workspace sparework;
    auto matrixT = new_matrix(work, nb, nb, sparework);

    // Options to forward
    auto&& larfbOpts = workspace_opts_t<void>{sparework};

    // Main loop
    for (idx_t i = i0; i != iN; i += inc) {
        idx_t ib = min<idx_t>(nb, k - i);
        const auto V = slice(A, pair{0, nA - k + i + ib}, pair{i, i + ib});
        const auto taui = slice(tau, pair{i, i + ib});
        auto matrixTi = slice(matrixT, pair{0, ib}, pair{0, ib});

        // Form the triangular factor of the block reflector
        // $H = H(i) H(i+1) ... H(i+ib-1)$
        larft(backward, columnwise_storage, V, taui, matrixTi);

        // H or H**H is applied to either C[i:m,0:n] or C[0:m,i:n]
        auto Ci = (side == Side::Left)
                      ? slice(C, pair{0, m - k + i + ib}, pair{0, n})
                      : slice(C, pair{0, m}, pair{0, n - k + i + ib});

        // Apply H or H**H
        larfb(side, trans, backward, columnwise_storage, V, matrixTi, Ci,
              larfbOpts);
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNMQL_HH
