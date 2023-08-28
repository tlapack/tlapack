/// @file geqrf.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgeqrf.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQRF_HH
#define TLAPACK_GEQRF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/geqr2.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"

namespace tlapack {
/**
 * Options struct for geqrf
 */
struct GeqrfOpts {
    size_t nb = 32;  ///< Block size
};

/** Worspace query of geqrf()
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
template <class T, TLAPACK_SMATRIX A_t, TLAPACK_SVECTOR tau_t>
constexpr WorkInfo geqrf_worksize(const A_t& A,
                                  const tau_t& tau,
                                  const GeqrfOpts& opts = {})
{
    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;
    using matrixT_t = matrix_type<A_t, tau_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = min((idx_t)opts.nb, k);

    auto&& A11 = cols(A, range(0, nb));
    auto&& tauw1 = slice(tau, range(0, nb));
    WorkInfo workinfo = geqr2_worksize<T>(A11, tauw1);

    if (n > nb) {
        auto&& TT1 = slice(A, range(0, nb), range(0, nb));
        auto&& A12 = slice(A, range(0, m), range(nb, n));
        workinfo.minMax(larfb_worksize<T>(LEFT_SIDE, CONJ_TRANS, FORWARD,
                                          COLUMNWISE_STORAGE, A11, TT1, A12));
        if constexpr (is_same_v<T, type_t<matrixT_t>>)
            workinfo += WorkInfo(nb, nb);
    }

    return workinfo;
}

/** Computes a QR factorization of an m-by-n matrix A using
 *  a blocked algorithm.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX A_t, TLAPACK_SVECTOR tau_t>
int geqrf(A_t& A, tau_t& tau, const GeqrfOpts& opts = {})
{
    Create<A_t> new_matrix;
    using T = type_t<A_t>;

    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = min((idx_t)opts.nb, k);

    // check arguments
    tlapack_check((idx_t)size(tau) >= k);

    // Allocate or get workspace
    WorkInfo workinfo = geqrf_worksize<T>(A, tau, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    auto TT = (n > nb) ? slice(work, range{workinfo.m - nb, workinfo.m},
                               range{workinfo.n - nb, workinfo.n})
                       : slice(work, range{0, 0}, range{0, 0});

    // Main computational loop
    for (idx_t j = 0; j < k; j += nb) {
        const idx_t ib = min(nb, k - j);

        // Compute the QR factorization of the current block A(j:m,j:j+ib)
        auto A11 = slice(A, range(j, m), range(j, j + ib));
        auto tauw1 = slice(tau, range(j, j + ib));

        geqr2_work(A11, tauw1, work);

        if (j + ib < n) {
            // Form the triangular factor of the block reflector H = H(j)
            // H(j+1) . . . H(j+ib-1)
            auto TT1 = slice(TT, range(0, ib), range(0, ib));
            larft(FORWARD, COLUMNWISE_STORAGE, A11, tauw1, TT1);

            // Apply H to A(j:m,j+ib:n) from the left
            auto A12 = slice(A, range(j, m), range(j + ib, n));
            larfb_work(LEFT_SIDE, CONJ_TRANS, FORWARD, COLUMNWISE_STORAGE, A11,
                       TT1, A12, work);
        }
    }

    return 0;
}
}  // namespace tlapack
#endif  // TLAPACK_GEQRF_HH
