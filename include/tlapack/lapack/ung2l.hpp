/// @file ung2l.hpp generates all or part of the unitary matrix Q from a QL
/// factorization determined by geqlf (unblocked algorithm).
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNG2L_HH
#define TLAPACK_UNG2L_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Worspace query of ung2l()
 *
 * @param[in] A m-by-n matrix.

 * @param[in] tau Real vector of length k. k <= n.
 *      The scalar factors of the elementary reflectors as returned by geqlf.
 *
 * @param[in] opts Options.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template <class matrix_t, class vector_t>
inline constexpr void ung2l_worksize(const matrix_t& A,
                                     const vector_t& tau,
                                     workinfo_t& workinfo,
                                     const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t n = ncols(A);

    auto v = col(A, n - 1);
    auto C = cols(A, range<idx_t>(0, n - 1));
    larf_worksize(left_side, backward, columnwise_storage, v, tau[n - 1], C,
                  workinfo, opts);
}

/**
 * @brief Generates a m-by-n matrix Q with orthonormal columns, which is defined
 * as the last n columns of a product of k elementary reflectors of order m
 * \[
 *     Q  =  H_{k-1} H_{k-2} ... H_0
 * \]
 * as returned by geqlf.
 *
 * @param[in,out] A m-by-n matrix. m >= n.
 *      On entry, the (n-k+i)-th column must contain the vector which defines
 * the elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by geqrf. On
 * exit, the m-by-n matrix $Q$.
 *
 * @param[in] tau Vector of length k. k <= n.
 *      The scalar factors of the elementary reflectors as returned by geqlf.
 *
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 *
 * @return 0 if success
 *
 * @ingroup computational
 */
template <class matrix_t, class vector_t>
int ung2l(matrix_t& A, const vector_t& tau, const workspace_opts_t<>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = size(tau);

    // check arguments
    tlapack_check(k <= n && n <= m);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        ung2l_worksize(A, tau, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{work};

    // Initialise columns 0:n-k to columns of the unit matrix
    for (idx_t j = 0; j < n - k; ++j) {
        for (idx_t l = 0; l < m; ++l)
            A(l, j) = zero;
        A(m - n + j, j) = one;
    }

    for (idx_t i = 0; i < k; ++i) {
        const idx_t j = n - k + i;

        // Define x, v and C
        auto x = slice(A, pair{0, m - n + j}, j);
        auto v = slice(A, pair{0, m - n + j + 1}, j);
        auto C = slice(A, pair{0, m - n + j + 1}, pair{0, j});

        // Apply $H_i$ to $A( 0:m-k+i+1, 0:n-k+i )$ from the left
        larf(left_side, backward, columnwise_storage, v, tau[i], C, larfOpts);
        scal(-tau[i], x);
        A(m - n + j, j) = one - tau[i];

        // Set A(m-k+i+1:m,n-k+i) to zero
        for (idx_t l = m - n + j + 1; l < m; ++l)
            A(l, j) = zero;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNG2L_HH
