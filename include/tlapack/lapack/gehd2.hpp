/// @file gehd2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dgehd2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEHD2_HH
#define TLAPACK_GEHD2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of gehd2()
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
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
inline constexpr WorkInfo gehd2_worksize(size_type<matrix_t> ilo,
                                         size_type<matrix_t> ihi,
                                         const matrix_t& A,
                                         const vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);

    WorkInfo workinfo;
    if (ilo + 1 < ihi && n > 0) {
        const auto v = slice(A, range{ilo + 1, ihi}, ilo);

        auto C0 = slice(A, range{0, ihi}, range{ilo + 1, ihi});
        workinfo = larf_worksize<T>(RIGHT_SIDE, FORWARD, COLUMNWISE_STORAGE, v,
                                    tau[0], C0);

        auto C1 = slice(A, range{ilo + 1, ihi}, range{ilo + 1, n});
        workinfo.minMax(larf_worksize<T>(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE,
                                         v, tau[0], C1));
    }

    return workinfo;
}

template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SMATRIX work_t>
int gehd2_work(size_type<matrix_t> ilo,
               size_type<matrix_t> ihi,
               matrix_t& A,
               vector_t& tau,
               work_t& work)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(ncols(A) != nrows(A));
    tlapack_check_false((idx_t)size(tau) < n - 1);

    // quick return
    if (n <= 0) return 0;

    for (idx_t i = ilo; i < ihi - 1; ++i) {
        // Define v := A[i+1:ihi,i]
        auto v = slice(A, range{i + 1, ihi}, i);

        // Generate the (i+1)-th elementary Householder reflection on v
        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau[i]);

        // Apply Householder reflection from the right to A[0:ihi,i+1:ihi]
        auto C0 = slice(A, range{0, ihi}, range{i + 1, ihi});
        larf_work(RIGHT_SIDE, FORWARD, COLUMNWISE_STORAGE, v, tau[i], C0, work);

        // Apply Householder reflection from the left to A[i+1:ihi,i+1:n-1]
        auto C1 = slice(A, range{i + 1, ihi}, range{i + 1, n});
        larf_work(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, v, conj(tau[i]), C1,
                  work);
    }

    return 0;
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
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int gehd2(size_type<matrix_t> ilo,
          size_type<matrix_t> ihi,
          matrix_t& A,
          vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;

    // functor
    Create<matrix_t> new_matrix;

    // constants
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(ncols(A) != nrows(A));
    tlapack_check_false((idx_t)size(tau) < n - 1);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = gehd2_worksize<T>(ilo, ihi, A, tau);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return gehd2_work(ilo, ihi, A, tau, work);
}

}  // namespace tlapack

#endif  // TLAPACK_GEHD2_HH
