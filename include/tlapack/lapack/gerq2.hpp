/// @file gerq2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgerq2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GERQ2_HH
#define TLAPACK_GERQ2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of gerq2()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau Not referenced.
 *
 * @param[in] opts Options.
 *
 * @return workinfo_t The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
inline constexpr workinfo_t gerq2_worksize(const matrix_t& A,
                                           const vector_t& tau,
                                           const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t n = ncols(A);

    if (n > 1) {
        auto C = cols(A, range<idx_t>{1, n});
        return larf_worksize(Side::Right, Direction::Backward, StoreV::Rowwise,
                             col(A, 0), tau[0], C, opts);
    }

    return workinfo_t{};
}

/** Computes an RQ factorization of a matrix A.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1' H_2' ... H_k',
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[n-k+i+1:n] = 0; v[n-k+i-1] = 1,
 * \]
 * with v[1] through v[n-k+i-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the m by n matrix A.
 *      On exit, if m <= n, the upper triangle of the subarray
 *      A(0:m,n-m:n) contains the m by m upper triangular matrix R;
 *      if m >= n, the elements on and above the (m-n)-th subdiagonal
 *      contain the m by n upper trapezoidal matrix R; the remaining
 *      elements, with the array TAU, represent the unitary matrix
 *      Q as a product of elementary reflectors.
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
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
int gerq2(matrix_t& A, vector_t& tau, const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>(m, n);

    // check arguments
    tlapack_check_false((idx_t)size(tau) < std::min<idx_t>(m, n));

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo = gerq2_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{work};

    for (idx_t i2 = 0; i2 < k; ++i2) {
        idx_t i = k - 1 - i2;

        // Define v := A[m-1-i2,0:n-i2]
        auto v = slice(A, m - 1 - i2, pair{0, n - i2});

        // Generate the (i+1)-th elementary Householder reflection on v
        larfg(Direction::Backward, StoreV::Rowwise, v, tau[i]);

        // Apply the reflector to the rest of the matrix
        if (m > i2 + 1) {
            auto C = slice(A, pair{0, m - 1 - i2}, pair{0, n - i2});
            larf(Side::Right, Direction::Backward, StoreV::Rowwise, v, tau[i],
                 C, larfOpts);
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GERQ2_HH
