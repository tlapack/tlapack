/// @file gelqf.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgelqf.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GELQF_HH
#define TLAPACK_GELQF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/gelq2.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"

namespace tlapack {
/**
 * Options struct for gelqf
 */
template <TLAPACK_INDEX idx_t = size_t>
struct GelqfOpts : public WorkspaceOpts {
    inline constexpr GelqfOpts(const WorkspaceOpts& opts = {})
        : WorkspaceOpts(opts){};

    idx_t nb = 32;  ///< Block size
};

/** Worspace query of gelqf()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau min(n,m) vector.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SMATRIX A_t, TLAPACK_SVECTOR tau_t>
inline constexpr WorkInfo gelqf_worksize(
    const A_t& A, const tau_t& tau, const GelqfOpts<size_type<A_t>>& opts = {})
{
    using idx_t = size_type<A_t>;
    using T = type_t<A_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = opts.nb;
    const idx_t ib = std::min<idx_t>(nb, k);

    auto A11 = rows(A, range(0, ib));
    auto TT1 = slice(A, range(0, ib), range(0, ib));
    auto A12 = slice(A, range(ib, m), range(0, n));
    auto tauw1 = slice(tau, range(0, ib));

    WorkInfo workinfo = gelq2_worksize(A11, tauw1);
    workinfo.minMax(larfb_worksize(Side::Right, Op::NoTrans, Direction::Forward,
                                   StoreV::Rowwise, A11, TT1, A12));

    workinfo += WorkInfo(sizeof(T) * nb, nb);

    return workinfo;
}

/** Computes an LQ factorization of an m-by-n matrix A using
 *  a blocked algorithm.
 *
 * The matrix Q is represented as a product of elementary reflectors.
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H,
 * \]
 * where k = min(m,n). Each H(j) has the form
 * \[
 *          H(j) = I - tauw * w * w**H
 * \]
 * where tauw is a scalar, and w is a vector with
 * \[
 *          w[0] = w[1] = ... = w[j-1] = 0; w[j] = 1,
 * \]
 * where w[j+1]**H through w[n]**H are stored on exit in the jth row of A.
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and below the diagonal of the array
 *      contain the m by min(m,n) lower trapezoidal matrix L (L is
 *      lower triangular if m <= n); the elements above the diagonal,
 *      with the array tauw, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] tau min(n,m) vector.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX A_t, TLAPACK_SVECTOR tau_t>
int gelqf(A_t& A, tau_t& tau, const GelqfOpts<size_type<A_t>>& opts = {})
{
    Create<A_t> new_matrix;

    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = opts.nb;

    // check arguments
    tlapack_check((idx_t)size(tau) >= k);

    // Allocate or get workspace
    VectorOfBytes localworkdata;
    Workspace work = [&]() {
        WorkInfo workinfo = gelqf_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    Workspace sparework;
    auto TT = new_matrix(work, nb, nb, sparework);

    // Options to forward
    const auto& gelq2Opts = WorkspaceOpts{sparework};
    const auto& larfbOpts = WorkspaceOpts{sparework};

    // Main computational loop
    for (idx_t j = 0; j < k; j += nb) {
        idx_t ib = std::min<idx_t>(nb, k - j);

        // Compute the LQ factorization of the current block A(j:j+ib-1,j:n)
        auto A11 = slice(A, range(j, j + ib), range(j, n));
        auto tauw1 = slice(tau, range(j, j + ib));

        gelq2(A11, tauw1, gelq2Opts);

        if (j + ib < k) {
            // Form the triangular factor of the block reflector H = H(j)
            // H(j+1) . . . H(j+ib-1)
            auto TT1 = slice(TT, range(0, ib), range(0, ib));
            larft(Direction::Forward, StoreV::Rowwise, A11, tauw1, TT1);

            // Apply H to A(j+ib:m,j:n) from the right
            auto A12 = slice(A, range(j + ib, m), range(j, n));
            larfb(Side::Right, Op::NoTrans, Direction::Forward, StoreV::Rowwise,
                  A11, TT1, A12, larfbOpts);
        }
    }

    return 0;
}
}  // namespace tlapack
#endif  // TLAPACK_GELQF_HH
