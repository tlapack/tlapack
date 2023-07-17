/// @file ung2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/ung2r.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNG2R_HH
#define TLAPACK_UNG2R_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Worspace query of ung2r()
 *
 * @param[in] A m-by-n matrix.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
inline constexpr WorkInfo ung2r_worksize(const matrix_t& A,
                                         const vector_t& tau,
                                         const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (n > 1 && m > 1) {
        auto C = cols(A, range{1, n});
        return larf_worksize(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, col(A, 0),
                             tau[0], C, opts);
    }
    return WorkInfo{};
}

/**
 * @brief Generates a matrix Q with orthogonal columns.
 * \[
 *     Q  =  H_1 H_2 ... H_k
 * \]
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the i-th column must contains the vector which defines the
 *      elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by geqrf.
 *      On exit, the m-by-n matrix $Q$.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 *
 * @return 0 if success
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int ung2r(matrix_t& A, const vector_t& tau, const WorkspaceOpts& opts = {})
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
    const idx_t k = size(tau);

    // check arguments
    tlapack_check_false(k > n);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    VectorOfBytes localworkdata;
    Workspace work = [&]() {
        WorkInfo workinfo = ung2r_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    const auto& larfOpts = WorkspaceOpts{work};

    // Initialise columns k:n-1 to columns of the unit matrix
    for (idx_t j = k; j < min(m, n); ++j) {
        for (idx_t l = 0; l < m; ++l)
            A(l, j) = zero;
        A(j, j) = one;
    }

    for (idx_t i = k - 1; i != idx_t(-1); --i) {
        // Apply $H_{i+1}$ to $A( i:m-1, i:n-1 )$ from the left
        // Define v and C
        auto v = slice(A, range{i, m}, i);
        auto C = slice(A, range{i, m}, range{i + 1, n});
        larf(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, v, tau[i], C, larfOpts);
        auto x = slice(A, range{i + 1, m}, i);
        scal(-tau[i], x);
        A(i, i) = one - tau[i];

        // Set A( 0:i-1, i ) to zero
        for (idx_t l = 0; l < i; l++)
            A(l, i) = zero;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNG2R_HH
