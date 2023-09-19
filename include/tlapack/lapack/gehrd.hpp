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
#include "tlapack/lapack/laset.hpp"

namespace tlapack {

/**
 * Options struct for gehrd
 */
struct GehrdOpts {
    size_t nb = 32;          ///< Block size used in the blocked reduction
    size_t nx_switch = 128;  ///< If only nx_switch columns are left, the
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
 * @param tau Vector of length n-1.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
constexpr WorkInfo gehrd_worksize(size_type<matrix_t> ilo,
                                  size_type<matrix_t> ihi,
                                  const matrix_t& A,
                                  const vector_t& tau,
                                  const GehrdOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = ncols(A);
    const idx_t nb = (ilo < ihi) ? min<idx_t>(opts.nb, ihi - ilo - 1) : 0;
    const idx_t nx = max<idx_t>(nb, opts.nx_switch);

    WorkInfo workinfo;
    if constexpr (is_same_v<T, type_t<work_t>>) {
        if (n > 0) {
            if ((ilo < ihi) && (nx < ihi - ilo - 1)) {
                workinfo = WorkInfo(n + nb, nb);

                auto&& V = slice(A, range{ilo + 1, ihi}, range{ilo, ilo + nb});
                auto&& T_s = slice(A, range{0, nb}, range{0, nb});
                auto&& A5 = slice(A, range{ilo + 1, ihi}, range{ilo + nb, n});

                workinfo.minMax(larfb_worksize<T>(LEFT_SIDE, CONJ_TRANS,
                                                  FORWARD, COLUMNWISE_STORAGE,
                                                  V, T_s, A5)
                                    .transpose());
            }
            workinfo.minMax(gehd2_worksize<T>(ilo, ihi, A, tau));
        }
    }

    return workinfo;
}

/** @copybrief gehrd()
 * Workspace is provided as an argument.
 * @copydetails gehrd()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int gehrd_work(size_type<matrix_t> ilo,
               size_type<matrix_t> ihi,
               matrix_t& A,
               vector_t& tau,
               work_t& work,
               const GehrdOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<work_t>;
    using range = pair<idx_t, idx_t>;
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;

    // constants
    const real_t one(1);
    const T zero(0);
    const idx_t n = ncols(A);

    // Blocksize
    const idx_t nb = (ilo < ihi) ? min<idx_t>(opts.nb, ihi - ilo - 1) : 0;
    // Size of the last block which be handled with unblocked code
    const idx_t nx = max(nb, (idx_t)opts.nx_switch);

    // check arguments
    tlapack_check_false((ilo < 0) or (ilo >= n));
    tlapack_check_false((ihi < 0) or (ihi > n));
    tlapack_check_false(ncols(A) != nrows(A));
    tlapack_check_false((idx_t)size(tau) < n - 1);

    // quick return
    if (n <= 0) return 0;

    // Reshape workspace
    WorkInfo workinfo = gehrd_worksize<T>(ilo, ihi, A, tau, opts);
    auto W = reshape(work, workinfo.m, workinfo.n);

    // Matrices Y and T
    auto Y = ((ilo < ihi) && (nx < ihi - ilo - 1))
                 ? slice(W, range{0, n}, range{0, nb})
                 : slice(W, range{0, 0}, range{0, 0});
    auto Yt = transpose_view(Y);
    auto matrixT = ((ilo < ihi) && (nx < ihi - ilo - 1))
                       ? slice(W, range{n, n + nb}, range{0, nb})
                       : slice(W, range{0, 0}, range{0, 0});
    laset(GENERAL, zero, zero, Y);

    idx_t i = ilo;
    for (; i + nx < ihi - 1; i = i + nb) {
        const idx_t nb2 = min(nb, ihi - i - 1);

        auto V = slice(A, range{i + 1, ihi}, range{i, i + nb2});
        auto A2 = slice(A, range{0, ihi}, range{i, ihi});
        auto tau2 = slice(tau, range{i, ihi});
        auto T_s = slice(matrixT, range{0, nb2}, range{0, nb2});
        auto Y_s = slice(Y, range{0, n}, range{0, nb2});
        lahr2(i, nb2, A2, tau2, T_s, Y_s);
        if (i + nb2 < ihi) {
            // Note, this V2 contains the last row of the triangular part
            auto V2 = slice(V, range{nb2 - 1, ihi - i - 1}, range{0, nb2});

            // Apply the block reflector H to A(0:ihi,i+nb:ihi) from the right,
            // computing A := A - Y * V**T. The multiplication requires
            // V(nb2-1,nb2-1) to be set to 1.
            const TA ei = V(nb2 - 1, nb2 - 1);
            V(nb2 - 1, nb2 - 1) = one;
            auto A3 = slice(A, range{0, ihi}, range{i + nb2, ihi});
            auto Y_2 = slice(Y, range{0, ihi}, range{0, nb2});
            gemm(NO_TRANS, CONJ_TRANS, -one, Y_2, V2, one, A3);
            V(nb2 - 1, nb2 - 1) = ei;
        }
        // Apply the block reflector H to A(0:i+1,i+1:i+ib) from the right
        auto V1 = slice(A, range{i + 1, i + nb2 + 1}, range{i, i + nb2});
        trmm(RIGHT_SIDE, LOWER_TRIANGLE, CONJ_TRANS, UNIT_DIAG, one, V1, Y_s);
        for (idx_t j = 0; j < nb2 - 1; ++j) {
            auto A4 = slice(A, range{0, i + 1}, i + j + 1);
            axpy(-one, slice(Y, range{0, i + 1}, j), A4);
        }

        // Apply the block reflector H to A(i+1:ihi,i+nb:n) from the left
        auto A5 = slice(A, range{i + 1, ihi}, range{i + nb2, n});
        larfb_work(LEFT_SIDE, CONJ_TRANS, FORWARD, COLUMNWISE_STORAGE, V, T_s,
                   A5, Yt);
    }

    return gehd2_work(i, ihi, A, tau, work);
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
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int gehrd(size_type<matrix_t> ilo,
          size_type<matrix_t> ihi,
          matrix_t& A,
          vector_t& tau,
          const GehrdOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;
    using T = type_t<work_t>;

    // Functor
    Create<work_t> new_matrix;

    // constants
    const idx_t n = ncols(A);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = gehrd_worksize<T>(ilo, ihi, A, tau, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return gehrd_work(ilo, ihi, A, tau, work, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_GEHRD_HH
