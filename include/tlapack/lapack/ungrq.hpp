/// @file ungrq.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zungrq.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGRQ_HH
#define TLAPACK_UNGRQ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/ungr2.hpp"

namespace tlapack {

/**
 * Options struct for ungrq
 */
template <class workT_t = void>
struct ungrq_opts_t : public workspace_opts_t<workT_t> {
    inline constexpr ungrq_opts_t(const workspace_opts_t<workT_t>& opts = {})
        : workspace_opts_t<workT_t>(opts){};

    size_type<workT_t> nb = 32;  ///< Block size
};

/** Worspace query of ungrq()
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
template <class matrix_t, class vector_t, class workT_t = void>
inline constexpr workinfo_t ungrq_worksize(
    const matrix_t& A,
    const vector_t& tau,
    const ungrq_opts_t<workT_t>& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using matrixT_t = deduce_work_t<workT_t, matrix_type<matrix_t, vector_t> >;
    using T = type_t<matrixT_t>;
    using pair = std::pair<idx_t, idx_t>;

    // Constants
    const idx_t k = size(tau);
    const idx_t nb = min<idx_t>(opts.nb, k);

    // Local workspace sizes
    workinfo_t workinfo(nb * sizeof(T), nb);

    // larfb:
    {
        // Constants
        const idx_t m = nrows(A);

        // Empty matrices
        const auto V = slice(A, pair{0, nb}, pair{0, m});
        const auto matrixT = slice(A, pair{0, nb}, pair{0, nb});

        // Internal workspace queries
        workinfo += larfb_worksize(right_side, conjTranspose, backward,
                                   rowwise_storage, V, matrixT, A, opts);
    }

    return workinfo;
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
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          class workT_t = void>
int ungrq(matrix_t& A,
          const vector_t& tau,
          const ungrq_opts_t<workT_t>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;
    using matrixT_t = deduce_work_t<workT_t, matrix_type<matrix_t, vector_t> >;

    // Functor
    Create<matrixT_t> new_matrix;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = size(tau);
    const idx_t nb = min<idx_t>(opts.nb, k);

    // check arguments
    tlapack_check_false(k > n);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo = ungrq_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Matrix T and recompute work
    Workspace sparework;
    auto matrixT = new_matrix(work, nb, nb, sparework);

    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{sparework};
    auto&& larfbOpts = workspace_opts_t<void>{sparework};

    // Initialise rows 0:m-k to rows of the unit matrix
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m - k; ++i)
            A(i, j) = zero;
        if (j + m >= n and j + k <= n) A(m + j - n, j) = one;
    }

    for (idx_t i = 0; i < k; i += nb) {
        idx_t ib = min<idx_t>(nb, k - i);
        idx_t ii = m - k + i;
        const auto taui = slice(tau, pair{i, i + ib});
        // Use block reflector to update most of the matrix
        // We do this first because the reflectors will be destroyed by the
        // unblocked code later.
        if (ii > 0) {
            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1)
            const auto V = slice(A, pair{ii, ii + ib}, pair{0, n - k + i + ib});
            auto matrixTi = slice(matrixT, pair{0, ib}, pair{0, ib});
            auto C = slice(A, pair{0, ii}, pair{0, n - k + i + ib});

            larft(backward, rowwise_storage, V, taui, matrixTi);
            larfb(right_side, conjTranspose, backward, rowwise_storage, V,
                  matrixTi, C, larfbOpts);
        }
        // Use unblocked code to apply H^H to columns 0:n-k+i+ib of current
        // block
        auto Ai = slice(A, pair{ii, ii + ib}, pair{0, n - k + i + ib});
        ungr2(Ai, taui, larfOpts);
        // Set columns n-k+i+ib:n of current block to zero
        for (idx_t j = n - k + i + ib; j < n; ++j)
            for (idx_t l = ii; l < ii + ib; l++)
                A(l, j) = zero;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNGRQ_HH
