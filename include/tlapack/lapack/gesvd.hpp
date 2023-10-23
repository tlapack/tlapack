/// @file gesvd.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zgesvd.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GESVD_HH
#define TLAPACK_GESVD_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/gebrd.hpp"
#include "tlapack/lapack/svd_qr.hpp"
#include "tlapack/lapack/ungbr.hpp"

namespace tlapack {

/**
 * Options struct for gesvd
 */
struct GesvdOpts {
    /// @todo If either max(m,n)/min(m,n) is larger than shapethresh, a QR
    /// factorization is used before
    float shapethresh = 1.6;
};

/**
 * Computes the singular values and, optionally, the right and/or
 * left singular vectors from the singular value decomposition (SVD) of
 * a real M-by-N matrix A. The SVD of A has the form
 *      B = U * S * V^H
 * where S is the diagonal matrix of singular values, U is a unitary
 * matrix of left singular vectors, and V is a unitary matrix of
 * right singular vectors. Depending on the dimensions of U and Vt,
 * either the reduced or full unitary factors are determined.
 *
 * @note There is no option to return U or Vt in A. This functionality is
 * present in zgesvd in Reference LAPACK.
 *
 * @return  0 if success
 *
 * @param[in] want_u bool
 *
 * @param[in] want_vt bool
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @param[out] s vector of length min(m,n).
 *      The singular values of A, sorted so that S(i) >= S(i+1).
 *
 * @param[in,out] U m-by-m matrix.
 *
 * @param[in,out] Vt n-by-n matrix.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR r_vector_t>
int gesvd(bool want_u,
          bool want_vt,
          matrix_t& A,
          r_vector_t& s,
          matrix_t& U,
          matrix_t& Vt,
          const GesvdOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Functors
    Create<vector_type<matrix_t>> new_vector;
    Create<vector_type<r_vector_t>> new_rvector;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const Uplo uplo = (m >= n) ? Uplo::Upper : Uplo::Lower;

    // Allocate vectors
    std::vector<type_t<matrix_t>> tauv_, tauw_;
    auto tauv = new_vector(tauv_, k);
    auto tauw = new_vector(tauw_, k);
    std::vector<type_t<r_vector_t>> e_;
    auto e = new_rvector(e_, k);

    // Reduce A to bidiagonal form
    gebrd(A, tauv, tauw);

    if (m >= n) {
        // copy upper bidiagonal matrix
        for (idx_t i = 0; i < k; ++i) {
            s[i] = real(A(i, i));
            if (i + 1 < n) e[i] = real(A(i, i + 1));
        }
    }
    else {
        // copy lower bidiagonal matrix
        for (idx_t i = 0; i < k; ++i) {
            s[i] = real(A(i, i));
            if (i + 1 < m) e[i] = real(A(i + 1, i));
        }
    }

    if (want_u) {
        auto Ui = slice(U, range{0, m}, range{0, k});
        lacpy(Uplo::Lower, slice(A, range{0, m}, range{0, k}), Ui);
        ungbr_q(n, U, tauv);
    }

    if (want_vt) {
        auto Vti = slice(Vt, range{0, k}, range{0, n});
        lacpy(Uplo::Upper, slice(A, range{0, k}, range{0, n}), Vti);
        ungbr_p(m, Vt, tauw);
    }

    return svd_qr(uplo, want_u, want_vt, s, e, U, Vt);
}

}  // namespace tlapack

#endif  // TLAPACK_GESVD_HH
