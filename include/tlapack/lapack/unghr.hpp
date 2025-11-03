/// @file unghr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dorghr.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGHR_HH
#define TLAPACK_UNGHR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ungq.hpp"

namespace tlapack {

/** Worspace query of unghr()
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      ilo and ihi must have the same values as in the
 *      previous call to gehrd. Q is equal to the unit
 *      matrix except in the submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *      0 <= ilo <= ihi <= max(1,n).
 *
 * @param[in] A m-by-n matrix.
 *
 * @param[in] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
constexpr WorkInfo unghr_worksize(size_type<matrix_t> ilo,
                                  size_type<matrix_t> ihi,
                                  const matrix_t& A,
                                  const vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t nh = (ihi > ilo + 1) ? ihi - 1 - ilo : 0;

    if (nh > 0 && ilo + 1 < ihi) {
        auto&& A_s = slice(A, range{ilo + 1, ihi}, range{ilo + 1, ihi});
        auto&& tau_s = slice(tau, range{ilo, ihi - 1});
        return ungq_worksize<T>(FORWARD, COLUMNWISE_STORAGE, A_s, tau_s);
    }
    return WorkInfo(0);
}

/** @copybrief unghr()
 * Workspace is provided as an argument.
 * @copydetails unghr()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int unghr_work(size_type<matrix_t> ilo,
               size_type<matrix_t> ihi,
               matrix_t& A,
               const vector_t& tau,
               work_t& work)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t nh = ihi > ilo + 1 ? ihi - 1 - ilo : 0;

    // check arguments
    tlapack_check_false((idx_t)size(tau) + 1 < min(m, n));

    // Shift the vectors which define the elementary reflectors one
    // column to the right, and set the first ilo and the last n-ihi
    // rows and columns to those of the unit matrix

    // This is currently optimised for column matrices, it may be interesting
    // to also write these loops for row matrices
    for (idx_t j = ihi - 1; j > ilo; --j) {
        for (idx_t i = 0; i < j; ++i) {
            A(i, j) = zero;
        }
        for (idx_t i = j + 1; i < ihi; ++i) {
            A(i, j) = A(i, j - 1);
        }
        for (idx_t i = ihi; i < n; ++i) {
            A(i, j) = zero;
        }
    }
    for (idx_t j = 0; j < ilo + 1; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            A(i, j) = zero;
        }
        A(j, j) = one;
    }
    for (idx_t j = ihi; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            A(i, j) = zero;
        }
        A(j, j) = one;
    }

    // Now that the vectors are shifted, we can call orgqr to generate the
    // matrix orgqr is not yet implemented, so we call org2r instead
    if (nh > 0) {
        auto A_s = slice(A, range{ilo + 1, ihi}, range{ilo + 1, ihi});
        auto tau_s = slice(tau, range{ilo, ihi - 1});
        return ungq_work(FORWARD, COLUMNWISE_STORAGE, A_s, tau_s, work);
    }
    else {
        return 0;
    }
}

/** Generates a m-by-n matrix Q with orthogonal columns.
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      ilo and ihi must have the same values as in the
 *      previous call to gehrd. Q is equal to the unit
 *      matrix except in the submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *      0 <= ilo <= ihi <= max(1,n).
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the vectors which define the elementary reflectors.
 *      On exit, the m-by-n matrix Q.
 *
 * @param[in] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int unghr(size_type<matrix_t> ilo,
          size_type<matrix_t> ihi,
          matrix_t& A,
          const vector_t& tau)
{
    using T = type_t<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // Allocates workspace
    WorkInfo workinfo = unghr_worksize<T>(ilo, ihi, A, tau);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return unghr_work(ilo, ihi, A, tau, work);
}

}  // namespace tlapack

#endif  // TLAPACK_UNGHR_HH
