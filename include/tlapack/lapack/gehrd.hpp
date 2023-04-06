/// @file gehrd.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dgehrd.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEHRD_HH
#define TLAPACK_GEHRD_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/gehd2.hpp"
#include "tlapack/lapack/lahr2.hpp"
#include "tlapack/lapack/larfb.hpp"

namespace tlapack {

/**
 * Options struct for gehrd
 */
template <class idx_t = size_t>
struct gehrd_opts_t : public workspace_opts_t<> {
    inline constexpr gehrd_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};

    idx_t nb = 32;          ///< Block size used in the blocked reduction
    idx_t nx_switch = 128;  ///< If only nx_switch columns are left, the
                            ///< algorithm will use unblocked code
};

/** Worspace query of gehrd()
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      It is assumed that A is already upper Hessenberg in columns
 *      0:ilo and rows ihi:n and is already upper triangular in
 *      columns ihi+1:n and rows 0:ilo.
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 * @param tau Not referenced.
 *
 * @param[in] opts Options.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
void gehrd_worksize(size_type<matrix_t> ilo,
                    size_type<matrix_t> ihi,
                    const matrix_t& A,
                    const vector_t& tau,
                    workinfo_t& workinfo,
                    const gehrd_opts_t<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;
    using T = type_t<work_t>;

    const idx_t n = ncols(A);
    const idx_t nb = std::min(opts.nb, ihi - ilo - 1);

    const workinfo_t myWorkinfo(sizeof(T) * (n + nb), nb);
    workinfo.minMax(myWorkinfo);
}

/** Reduces a general square matrix to upper Hessenberg form
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_ilo H_ilo+1 ... H_ihi,
 * \]
 * Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i] = 0; v[i+1] = 1,
 * \]
 * with v[i+2] through v[ihi] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      It is assumed that A is already upper Hessenberg in columns
 *      0:ilo and rows ihi:n and is already upper triangular in
 *      columns ihi+1:n and rows 0:ilo.
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in,out] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 *      On exit, the upper triangle and the first subdiagonal of A
 *      are overwritten with the upper Hessenberg matrix H, and the
 *      elements below the first subdiagonal, with the array TAU,
 *      represent the orthogonal matrix Q as a product of elementary
 *      reflectors. See Further Details.
 * @param[out] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
int gehrd(size_type<matrix_t> ilo,
          size_type<matrix_t> ihi,
          matrix_t& A,
          vector_t& tau,
          const gehrd_opts_t<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;
    using pair = pair<idx_t, idx_t>;
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;

    // Functor
    Create<work_t> new_matrix;

    // constants
    const real_t one(1);
    const type_t<work_t> zero(0);
    const idx_t n = ncols(A);

    // Blocksize
    idx_t nb = std::min(opts.nb, ihi - ilo - 1);
    // Size of the last block which be handled with unblocked code
    idx_t nx_switch = opts.nx_switch;
    idx_t nx = std::max(nb, nx_switch);

    // check arguments
    tlapack_check_false((ilo < 0) or (ilo >= n));
    tlapack_check_false((ihi < 0) or (ihi > n));
    tlapack_check_false(ncols(A) != nrows(A));
    tlapack_check_false((idx_t)size(tau) < n - 1);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        gehrd_worksize(ilo, ihi, A, tau, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfbOpts = workspace_opts_t<transpose_type<work_t> >{work};
    auto&& gehd2Opts = workspace_opts_t<>{work};

    // Matrix Y
    Workspace workMatrixT;
    auto Y = new_matrix(work, n, nb, workMatrixT);
    laset(dense, zero, zero, Y);

    // Matrix T
    auto matrixT = new_matrix(workMatrixT, nb, nb);

    idx_t i = ilo;
    for (; i + nx < ihi - 1; i = i + nb) {
        const idx_t nb2 = std::min(nb, ihi - i - 1);

        auto V = slice(A, pair{i + 1, ihi}, pair{i, i + nb2});
        auto A2 = slice(A, pair{0, ihi}, pair{i, ihi});
        auto tau2 = slice(tau, pair{i, ihi});
        auto T_s = slice(matrixT, pair{0, nb2}, pair{0, nb2});
        auto Y_s = slice(Y, pair{0, n}, pair{0, nb2});
        lahr2(i, nb2, A2, tau2, T_s, Y_s);
        if (i + nb2 < ihi) {
            // Note, this V2 contains the last row of the triangular part
            auto V2 = slice(V, pair{nb2 - 1, ihi - i - 1}, pair{0, nb2});

            // Apply the block reflector H to A(0:ihi,i+nb:ihi) from the right,
            // computing A := A - Y * V**T. The multiplication requires
            // V(nb2-1,nb2-1) to be set to 1.
            const TA ei = V(nb2 - 1, nb2 - 1);
            V(nb2 - 1, nb2 - 1) = one;
            auto A3 = slice(A, pair{0, ihi}, pair{i + nb2, ihi});
            auto Y_2 = slice(Y, pair{0, ihi}, pair{0, nb2});
            gemm(Op::NoTrans, Op::ConjTrans, -one, Y_2, V2, one, A3);
            V(nb2 - 1, nb2 - 1) = ei;
        }
        // Apply the block reflector H to A(0:i+1,i+1:i+ib) from the right
        auto V1 = slice(A, pair{i + 1, i + nb2 + 1}, pair{i, i + nb2});
        trmm(Side::Right, Uplo::Lower, Op::ConjTrans, Diag::Unit, one, V1, Y_s);
        for (idx_t j = 0; j < nb2 - 1; ++j) {
            auto A4 = slice(A, pair{0, i + 1}, i + j + 1);
            axpy(-one, slice(Y, pair{0, i + 1}, j), A4);
        }

        // Apply the block reflector H to A(i+1:ihi,i+nb:n) from the left
        auto A5 = slice(A, pair{i + 1, ihi}, pair{i + nb2, n});
        larfb(Side::Left, Op::ConjTrans, Direction::Forward, StoreV::Columnwise,
              V, T_s, A5, larfbOpts);
    }

    gehd2(i, ihi, A, tau, gehd2Opts);

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GEHRD_HH
