/// @file geql2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgeql2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQL2_HH
#define TLAPACK_GEQL2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of geql2()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau Not referenced.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
inline constexpr WorkInfo geql2_worksize(const matrix_t& A,
                                         const vector_t& tau,
                                         const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);

    if (n > 1) {
        auto C = cols(A, range{1, n});
        return larf_worksize(Side::Left, Direction::Backward,
                             StoreV::Columnwise, col(A, 0), tau[0], C, opts);
    }

    return WorkInfo{};
}

/** Computes a QL factorization of a matrix A.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_k ... H_2 H_1,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[m-k+i+1:m] = 0; v[m-k+i-1] = 1,
 * \]
 * with v[1] through v[n-k+i-1] stored on exit below the diagonal
 * in A(0:m-k+i-1,n-k+i), and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the m by n matrix A.
 *      On exit, if m >= n, the lower triangle of A(m-n:m,0:n) contains
 *      the n by n lower triangular matrix L;
 *      If m <= n, the elements on and below the (n-m)-th
 *      superdiagonal contain the m by n lower trapezoidal matrix L
 *      the remaining elements, with the array TAU, represent the
 *      unitary matrix Q as a product of elementary reflectors.
 *
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int geql2(matrix_t& A, vector_t& tau, const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>(m, n);

    // check arguments
    tlapack_check_false((idx_t)size(tau) < std::min<idx_t>(m, n));

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    VectorOfBytes localworkdata;
    Workspace work = [&]() {
        WorkInfo workinfo = geql2_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    const auto& larfOpts = WorkspaceOpts{work};

    for (idx_t i2 = 0; i2 < k; ++i2) {
        idx_t i = k - 1 - i2;

        // Column to be reduced
        auto v = slice(A, range{0, m - k + i + 1}, n - k + i);

        // Generate the (i+1)-th elementary Householder reflection on v
        larfg(Direction::Backward, StoreV::Columnwise, v, tau[i]);

        // Apply the reflector to the rest of the matrix
        if (n + i > k) {
            auto C = slice(A, range{0, m - k + i + 1}, range{0, n - k + i});
            larf(Side::Left, Direction::Backward, StoreV::Columnwise, v,
                 conj(tau[i]), C, larfOpts);
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GEQL2_HH
