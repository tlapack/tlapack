/// @file gebrd.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zgebrd.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEBRD_HH
#define TLAPACK_GEBRD_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/gebd2.hpp"
#include "tlapack/lapack/labrd.hpp"
#include "tlapack/lapack/larfb.hpp"

namespace tlapack {

/**
 * Options struct for gebrd
 */
template <TLAPACK_INDEX idx_t = size_t>
struct gebrd_opts_t : public workspace_opts_t<> {
    inline constexpr gebrd_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};

    idx_t nb = 32;  ///< Block size used in the blocked reduction
};

/** Worspace query of gebrd()
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @param[out] d Real vector of length min(m,n).
 *      Diagonal elements of B
 *
 * @param[out] e Real vector of length min(m,n).
 *      Off-diagonal elements of B
 *
 * @param[out] tauq vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix Q.
 *
 * @param[out] taup vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix P.
 *
 * @return workinfo_t The amount workspace required.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_VECTOR r_vector_t>
workinfo_t gebrd_worksize(const matrix_t& A,
                          r_vector_t& d,
                          r_vector_t& e,
                          const vector_t& tauq,
                          const vector_t& taup,
                          const gebrd_opts_t<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;
    using T = type_t<work_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t nb = min(opts.nb, min(m, n));

    return workinfo_t(sizeof(T) * (m + n), nb);
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
 *      On exit, if m >= n, the diagonal and the first superdiagonal
 *      are overwritten with the upper bidiagonal matrix B; the
 *      elements below the diagonal, with the array tauv, represent
 *      the unitary matrix Q as a product of elementary reflectors,
 *      and the elements above the first superdiagonal, with the array
 *      tauw, represent the unitary matrix P as a product of elementary
 *      reflectors.
 *
 * @param[out] d Real vector of length min(m,n).
 *      Diagonal elements of B
 *
 * @param[out] e Real vector of length min(m,n).
 *      Off-diagonal elements of B
 *
 * @param[out] tauq vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix Q.
 *
 * @param[out] taup vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix P.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_VECTOR r_vector_t>
int gebrd(matrix_t& A,
          r_vector_t& d,
          r_vector_t& e,
          vector_t& tauq,
          vector_t& taup,
          const gebrd_opts_t<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;
    using pair = pair<idx_t, idx_t>;
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;

    // Functor
    Create<work_t> new_matrix;

    // constants
    const real_t one(1);
    const type_t<work_t> zero(0);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = min(opts.nb, k);

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo = gebrd_worksize(A, d, e, tauq, taup, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Matrix X
    Workspace workMatrixY;
    auto X = new_matrix(work, m, nb, workMatrixY);
    laset(dense, zero, zero, X);

    // Matrix Y
    Workspace spareWork;
    auto Y = new_matrix(workMatrixY, n, nb, spareWork);
    laset(dense, zero, zero, Y);

    for (idx_t i = 0; i < k; i = i + nb) {
        idx_t ib = min(nb, k - i);
        // Reduce rows and columns i:i+ib-1 to bidiagonal form and return
        // the matrices X and Y which are needed to update the unreduced
        // part of the matrix
        auto A2 = slice(A, pair{i, m}, pair{i, n});
        auto d2 = slice(d, pair{i, i + ib});
        auto e2 = slice(e, pair{i, i + ib});
        auto tauq2 = slice(tauq, pair{i, i + ib});
        auto taup2 = slice(taup, pair{i, i + ib});
        auto X2 = slice(X, pair{i, m}, pair{0, ib});
        auto Y2 = slice(Y, pair{i, n}, pair{0, ib});
        labrd(A2, d2, e2, tauq2, taup2, X2, Y2);

        //
        // Update the trailing submatrix A(i+nb:m,i+nb:n), using an update
        // of the form  A := A - V*Y**H - X*U**H
        //
        auto A3 = slice(A, pair{i + ib, m}, pair{i + ib, n});
        auto V = slice(A, pair{i + ib, m}, pair{i, i + ib});
        auto Y3 = slice(Y, pair{i + ib, n}, pair{0, ib});
        gemm(noTranspose, conjTranspose, -one, V, Y3, one, A3);
        auto U = slice(A, pair{i, i + ib}, pair{i + ib, n});
        auto X3 = slice(X, pair{i + ib, m}, pair{0, ib});
        gemm(noTranspose, noTranspose, -one, X3, U, one, A3);

        //
        // Copy diagonal and off-diagonal elements of B back into A
        //
        if (m >= n) {
            // copy upper bidiagonal matrix
            for (idx_t j = i; j < i + ib; ++j) {
                A(j, j) = d[j];
                if (j + 1 < n) A(j, j + 1) = e[j];
            }
        }
        else {
            // copy lower bidiagonal matrix
            for (idx_t j = i; j < i + ib; ++j) {
                A(j, j) = d[j];
                if (j + 1 < m) A(j + 1, j) = e[j];
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GEBRD_HH
