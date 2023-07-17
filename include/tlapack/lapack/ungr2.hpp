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
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
inline constexpr WorkInfo ungr2_worksize(const matrix_t& A,
                                         const vector_t& tau,
                                         const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);

    if (m > 1) {
        auto C = rows(A, range{1, m});
        return larf_worksize(RIGHT_SIDE, BACKWARD, ROWWISE_STORAGE, row(A, 0),
                             tau[0], C, opts);
    }
    return WorkInfo{};
}

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last m rows of a product of k elementary
 *        reflectors of order n
 * \[
 *     Q  =  H_1^H H_2^H ... H_k^H
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
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int ungr2(matrix_t& A, const vector_t& tau, const WorkspaceOpts& opts = {})
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
    tlapack_check_false(k < 0 || k > n);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    VectorOfBytes localworkdata;
    Workspace work = [&]() {
        WorkInfo workinfo = ungr2_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    const auto& larfOpts = WorkspaceOpts{work};

    // Initialise rows 0:m-k to rows of the unit matrix
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m - k; ++i)
            A(i, j) = zero;
        if (j + m >= n and j + k <= n) A(m + j - n, j) = one;
    }

    for (idx_t i = 0; i < k; ++i) {
        idx_t ii = m - k + i;
        auto v = slice(A, ii, range{0, n - k + 1 + i});
        auto C = slice(A, range{0, ii}, range{0, n - k + 1 + i});
        larf(Side::Right, Direction::Backward, StoreV::Rowwise, v, conj(tau[i]),
             C, larfOpts);
        auto x = slice(A, ii, range{0, n - k + i});
        scal(-conj(tau[i]), x);
        A(ii, n - k + i) = one - conj(tau[i]);

        // Set A( 0:i-1, i ) to zero
        for (idx_t l = n - k + i + 1; l < n; l++)
            A(ii, l) = zero;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNGR2_HH
