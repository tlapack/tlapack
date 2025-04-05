/// @file gebrd.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zgebrd.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEBRD_HH
#define TLAPACK_GEBRD_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/labrd.hpp"

namespace tlapack {

/**
 * Options struct for gebrd()
 */
struct GebrdOpts {
    size_t nb = 32;  ///< Block size used in the blocked reduction
};

/** Worspace query of gebrd()
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @param[out] tauq vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix Q.
 *
 * @param[out] taup vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix P.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @param[in] opts Options.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
constexpr WorkInfo gebrd_worksize(const matrix_t& A,
                                  const vector_t& tauq,
                                  const vector_t& taup,
                                  const GebrdOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t nb = min((idx_t)opts.nb, min(m, n));

    if constexpr (is_same_v<T, type_t<work_t>>)
        return WorkInfo(m + n, nb);
    else
        return WorkInfo(0);
}

/** @copybrief gebrd()
 * Workspace is provided as an argument.
 * @copydetails gebrd()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int gebrd_work(matrix_t& A,
               vector_t& tauv,
               vector_t& tauw,
               work_t& work,
               const GebrdOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;

    // constants
    const real_t one(1);
    const type_t<work_t> zero(0);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = min((idx_t)opts.nb, k);

    // Matrices X and Y
    auto [X, work2] = reshape(work, m, nb);
    auto [Y, work3] = reshape(work2, n, nb);
    laset(GENERAL, zero, zero, X);
    laset(GENERAL, zero, zero, Y);

    for (idx_t i = 0; i < k; i = i + nb) {
        idx_t ib = min(nb, k - i);
        // Reduce rows and columns i:i+ib-1 to bidiagonal form and return
        // the matrices X and Y which are needed to update the unreduced
        // part of the matrix
        auto A2 = slice(A, range{i, m}, range{i, n});
        auto tauq = slice(tauv, range{i, i + ib});
        auto taup = slice(tauw, range{i, i + ib});
        auto X2 = slice(X, range{i, m}, range{0, ib});
        auto Y2 = slice(Y, range{i, n}, range{0, ib});
        labrd(A2, tauq, taup, X2, Y2);

        //
        // Update the trailing submatrix A(i+nb:m,i+nb:n), using an update
        // of the form  A := A - V*Y**H - X*U**H
        //
        if (i + ib < m && i + ib < n) {
            real_t e;
            auto A3 = slice(A, range{i + ib, m}, range{i + ib, n});

            if (m >= n) {
                e = real(A(i + ib - 1, i + ib));
                A(i + ib - 1, i + ib) = one;
            }
            else {
                e = real(A(i + ib, i + ib - 1));
                A(i + ib, i + ib - 1) = one;
            }

            auto V = slice(A, range{i + ib, m}, range{i, i + ib});
            auto Y3 = slice(Y, range{i + ib, n}, range{0, ib});
            gemm(NO_TRANS, CONJ_TRANS, -one, V, Y3, one, A3);
            auto U = slice(A, range{i, i + ib}, range{i + ib, n});
            auto X3 = slice(X, range{i + ib, m}, range{0, ib});
            gemm(NO_TRANS, NO_TRANS, -one, X3, U, one, A3);

            if (m >= n)
                A(i + ib - 1, i + ib) = e;
            else
                A(i + ib, i + ib - 1) = e;
        }
    }

    return 0;
}

/** Reduces a general m by n matrix A to an upper
 *  real bidiagonal form B by a unitary transformation:
 * \[
 *          Q**H * A * P = B,
 * \]
 *  where m >= n.
 *
 * The matrices Q and P are represented as products of elementary
 * reflectors:
 *
 * If m >= n,
 * \[
 *          Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
 * \]
 * Each H(i) and G(i) has the form:
 * \[
 *          H(j) = I - tauv * v * v**H  and G(j) = I - tauw * w * w**H
 * \]
 * where tauv and tauw are scalars, and v and w are
 * vectors; v(1:j-1) = 0, v(j) = 1, and v(j+1:m) is stored on exit in
 * A(j+1:m,j); w(1:j) = 0, w(j+1) = 1, and w(j+2:n) is stored on exit in
 * A(j,i+2:n); tauv is stored in tauv(j) and tauw in tauw(j).
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the m by n general matrix to be reduced.
 *      On exit,
 *      - if m >= n, the diagonal and the first superdiagonal
 *        are overwritten with the upper bidiagonal matrix B; the
 *        elements below the diagonal, with the array tauv, represent
 *        the unitary matrix Q as a product of elementary reflectors,
 *        and the elements above the first superdiagonal, with the array
 *        tauw, represent the unitary matrix P as a product of elementary
 *        reflectors.
 *      - if m < n, the diagonal and the first superdiagonal
 *        are overwritten with the lower bidiagonal matrix B; the
 *        elements below the first subdiagonal, with the array tauv, represent
 *        the unitary matrix Q as a product of elementary reflectors,
 *        and the elements above the diagonal, with the array tauw, represent
 *        the unitary matrix P as a product of elementary reflectors.
 *
 * @param[out] tauv vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix Q.
 *
 * @param[out] tauw vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix P.
 *
 * @param[in] opts Options.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int gebrd(matrix_t& A,
          vector_t& tauv,
          vector_t& tauw,
          const GebrdOpts& opts = {})
{
    using work_t = matrix_type<matrix_t, vector_t>;
    using T = type_t<work_t>;

    // Functor
    Create<work_t> new_matrix;

    // Allocates workspace
    WorkInfo workinfo = gebrd_worksize<T>(A, tauv, tauw, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return gebrd_work(A, tauv, tauw, work, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_GEBRD_HH
