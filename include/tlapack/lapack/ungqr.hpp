/// @file ungqr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zungqr.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGQR_HH
#define TLAPACK_UNGQR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/ung2r.hpp"

namespace tlapack {

/**
 * Options struct for ungqr
 */
template <TLAPACK_INDEX idx_t = size_t>
struct UngqrOpts : public WorkspaceOpts {
    inline constexpr UngqrOpts(const WorkspaceOpts& opts = {})
        : WorkspaceOpts(opts){};

    idx_t nb = 32;  ///< Block size
};

/** Worspace query of ungqr()
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
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
inline constexpr WorkInfo ungqr_worksize(
    const matrix_t& A,
    const vector_t& tau,
    const UngqrOpts<size_type<matrix_t>>& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using matrixT_t = matrix_type<matrix_t, vector_t>;
    using T = type_t<matrixT_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const idx_t k = size(tau);
    const idx_t nb = min<idx_t>(opts.nb, k);

    // Local workspace sizes
    WorkInfo workinfo(nb * sizeof(T), nb);

    // larfb:
    {
        // Constants
        const idx_t m = nrows(A);

        // Empty matrices
        const auto V = slice(A, range{0, m}, range{0, nb});
        const auto matrixT = slice(A, range{0, nb}, range{0, nb});

        // Internal workspace queries
        workinfo += larfb_worksize(LEFT_SIDE, NO_TRANS, FORWARD,
                                   COLUMNWISE_STORAGE, V, matrixT, A, opts);
    }

    return workinfo;
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
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int ungqr(matrix_t& A,
          const vector_t& tau,
          const UngqrOpts<size_type<matrix_t>>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using matrixT_t = matrix_type<matrix_t, vector_t>;

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
    VectorOfBytes localworkdata;
    Workspace work = [&]() {
        WorkInfo workinfo = ungqr_worksize(A, tau, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Matrix T and recompute work
    Workspace sparework;
    auto matrixT = new_matrix(work, nb, nb, sparework);

    // Options to forward
    const auto& larfOpts = WorkspaceOpts{sparework};
    const auto& larfbOpts = WorkspaceOpts{sparework};

    // Initialise columns k:n-1 to columns of the unit matrix
    for (idx_t j = k; j < min(m, n); ++j) {
        for (idx_t l = 0; l < m; ++l)
            A(l, j) = zero;
        A(j, j) = one;
    }

    for (idx_t i = ((k - 1) / nb) * nb; i != idx_t(-nb); i = i - nb) {
        idx_t ib = min<idx_t>(nb, k - i);
        const auto taui = slice(tau, range{i, i + ib});
        // Use block reflector to update most of the matrix
        // We do this first because the reflectors will be destroyed by the
        // unblocked code later.
        if (i + ib < n) {
            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1)
            const auto V = slice(A, range{i, m}, range{i, i + ib});
            auto matrixTi = slice(matrixT, range{0, ib}, range{0, ib});
            auto C = slice(A, range{i, m}, range{i + ib, n});

            larft(FORWARD, COLUMNWISE_STORAGE, V, taui, matrixTi);
            larfb(LEFT_SIDE, NO_TRANS, FORWARD, COLUMNWISE_STORAGE, V, matrixTi,
                  C, larfbOpts);
        }
        // Use unblocked code to apply H to rows i:m of current block
        auto Ai = slice(A, range{i, m}, range{i, i + ib});
        ung2r(Ai, taui, larfOpts);
        // Set rows 0:i-1 of current block to zero
        for (idx_t j = i; j < i + ib; ++j)
            for (idx_t l = 0; l < i; l++)
                A(l, j) = zero;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNGQR_HH
