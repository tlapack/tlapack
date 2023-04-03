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
template <class idx_t = size_t>
struct gebrd_opts_t : public workspace_opts_t<> {
    inline constexpr gebrd_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};

    idx_t nb = 32;          ///< Block size used in the blocked reduction
    idx_t nx_switch = 128;  ///< If only nx_switch columns are left, the
                            ///< algorithm will use unblocked code
};

/** Worspace query of gebrd()
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      It is assumed that A is already upper Hessenberg in columns
 *      0:ilo and rows ihi:n and is already upper triangular in
 *      columns ihi+1:n and rows 0:ilo.
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 * @param tau Not referenced.
 *
 * @param[in] opts Options.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template <class matrix_t, class vector_t, class r_vector_t>
void gebrd_worksize(const matrix_t& A,
                    r_vector_t& d,
                    r_vector_t& e,
                    const vector_t& tauq,
                    const vector_t& taup,
                    workinfo_t& workinfo,
                    const gebrd_opts_t<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, vector_t>;
    using T = type_t<work_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t nb = min(opts.nb, min(m, n));

    const workinfo_t myWorkinfo(sizeof(T) * (m + n), nb);
    workinfo.minMax(myWorkinfo);
}

template <class matrix_t, class vector_t, class r_vector_t>
int gebrd(const matrix_t& A,
          r_vector_t& d,
          r_vector_t& e,
          const vector_t& tauq,
          const vector_t& taup,
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
        workinfo_t workinfo;
        gebrd_worksize(A, d, e, tauq, taup, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfbOpts = workspace_opts_t<transpose_type<work_t> >{work};
    auto&& gehd2Opts = workspace_opts_t<>{work};

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
