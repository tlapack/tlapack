/// @file ungr2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zungr2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGR2_HH
#define TLAPACK_UNGR2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Worspace query of ungr2()
 *
 *
 * @param[in] A m-by-n matrix.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
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
inline constexpr void ungr2_worksize(const matrix_t& A,
                                     const vector_t& tau,
                                     workinfo_t& workinfo,
                                     const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t n = ncols(A);

    if (n > 1) {
        auto C = cols(A, range<idx_t>{1, n});
        larf_worksize(left_side, forward, columnwise_storage, col(A, 0), tau[0],
                      C, workinfo, opts);
    }
}

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last m rows of a product of k elementary
 *        reflectors of order n
 * \[
 *     Q  =  H_1' H_2' ... H_k'
 * \]
 *        The reflectors are stored in the matrix A as returned by gerqf
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the (m-k+i)-th row must contain the vector which
 *      defines the elementary reflector H(i), for i = 1,2,...,k, as
 *      returned by GERQF in the last k rows of its matrix argument A.
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
template <class matrix_t, class vector_t>
int ungr2(matrix_t& A, const vector_t& tau, const workspace_opts_t<>& opts = {})
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
    tlapack_check_false(k < 0 || k > n);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        ungr2_worksize(A, tau, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{work};

    // Initialise rows 0:m-k to rows of the unit matrix
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m - k; ++i)
            A(i, j) = zero;
        if (j + m >= n and j + k <= n) A(m + j - n, j) = one;
    }

    for (idx_t i = 0; i < k; ++i) {
        idx_t ii = m - k + i;
        if (ii > 0) {
            auto v = slice(A, ii, pair{0, n - k + 1 + i});
            auto C = slice(A, pair{0, ii}, pair{0, n - k + 1 + i});
            larf(Side::Right, Direction::Backward, StoreV::Rowwise, v,
                 conj(tau[i]), C, larfOpts);
        }
        if (n + i > k) {
            auto x = slice(A, ii, pair{0, n - k + i});
            scal(-conj(tau[i]), x);
        }
        A(ii, n - k + i) = one - conj(tau[i]);

        // Set A( 0:i-1, i ) to zero
        for (idx_t l = n - k + i + 1; l < n; l++)
            A(ii, l) = zero;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNGR2_HH
