/// @file ung2l.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zung2l.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNG2L_HH
#define TLAPACK_UNG2L_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Worspace query of ung2l()
 *
 *
 * @param[in] A m-by-n matrix.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @return workinfo_t The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class matrix_t, class vector_t>
inline constexpr workinfo_t ung2l_worksize(const matrix_t& A,
                                           const vector_t& tau,
                                           const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t n = ncols(A);

    if (n > 1) {
        auto C = cols(A, range<idx_t>{1, n});
        return larf_worksize(Side::Left, Direction::Backward,
                             StoreV::Columnwise, col(A, 0), tau[0], C, opts);
    }
    return workinfo_t{};
}

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last n columns of a product of k elementary
 *        reflectors of order m
 * \[
 *     Q  =  H_k ... H_2 H_1
 * \]
 *        The reflectors are stored in the matrix A as returned by geqlf
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the (n+k-i)-th column must contains the vector which defines
 the
 *      elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by geqlf.
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
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
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
    tlapack_check_false(k < 0 || k > n);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo = ung2l_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{work};

    // Initialise rows 0:m-k to rows of the unit matrix
    for (idx_t j = 0; j < n - k; ++j) {
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = zero;
        A(m - n + j, j) = one;
    }

    for (idx_t i = 0; i < k; ++i) {
        idx_t ii = n - k + i;
        auto v = slice(A, pair{0, m - k + i + 1}, ii);
        auto C = slice(A, pair{0, m - k + i + 1}, pair{0, ii});
        larf(Side::Left, Direction::Backward, StoreV::Columnwise, v, tau[i], C,
             larfOpts);
        auto x = slice(A, pair{0, m - k + i}, ii);
        scal(-tau[i], x);
        A(m - k + i, ii) = one - tau[i];

        // Set A( 0:i-1, i ) to zero
        for (idx_t l = m - k + i + 1; l < m; l++)
            A(l, ii) = zero;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNG2L_HH
