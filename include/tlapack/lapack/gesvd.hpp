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

#include <optional>

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/gebrd.hpp"
#include "tlapack/lapack/svd_qr.hpp"
#include "tlapack/lapack/ungbr.hpp"

namespace tlapack {

/**
 * Options struct for gesvd
 */
struct GesvdOpts {
    // If either max(m,n)/min(m,n) is larger than shapethresh, a QR
    // factorization is used before
    float shapethresh = 1.6;
};

/** Worspace query of gesvd()
 *
 * @return The amount of workspace required.
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
 * @ingroup workspace_query
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR r_vector_t>
constexpr WorkInfo gesvd_worksize(bool want_u,
                                  bool want_vt,
                                  matrix_t& A,
                                  r_vector_t& s,
                                  matrix_t& U,
                                  matrix_t& Vt,
                                  const GesvdOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    // using pair = std::pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);

    WorkInfo workinfo(k, 3);

    // if (m >= n) {
    //     auto tau = slice(A, 0, pair{0, k});
    //     workinfo += gebrd_worksize(A, tau, tau);
    //     if (want_u) {
    //         workinfo += ungbr_q_worksize(k, n, U, tau);
    //     }
    //     if (want_vt) {
    //         workinfo += ungbr_p_worksize(k, m, Vt, tau);
    //     }
    // }
    // else {
    //     auto tau = slice(A, pair{0, k}, 0);
    //     workinfo += gebrd_worksize(A, tau, tau);
    //     if (want_u) {
    //         workinfo += ungbr_q_worksize(k, n, U, tau);
    //     }
    //     if (want_vt) {
    //         workinfo += ungbr_p_worksize(k, m, Vt, tau);
    //     }
    // }

    return workinfo;
}

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
 * NOTE: the LAPACK function GESVD also allows returning either U or Vt
 * inside of A. I'm not sure how to design the interface to allow this.
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
    using work_t = matrix_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<work_t> new_matrix;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);

    // Allocates workspace
    WorkInfo workinfo = gesvd_worksize(want_u, want_vt, A, s, U, Vt, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    auto tauv = col(work, 1);
    auto tauw = col(work, 2);

    gebrd(A, tauv, tauw, GebrdOpts{});

    // For now, we use a locally allocated vector, because I don't know how to
    // make a real_t vector wrapper of the correct type.
    std::vector<real_t> e(k);
    // auto e = col(work, 0);

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
        auto Ui = slice(U, pair{0, m}, pair{0, k});
        lacpy(Uplo::Lower, slice(A, pair{0, m}, pair{0, k}), Ui);
        ungbr_q(n, U, tauv, UngbrOpts{});
    }

    if (want_vt) {
        auto Vti = slice(Vt, pair{0, k}, pair{0, n});
        lacpy(Uplo::Upper, slice(A, pair{0, k}, pair{0, n}), Vti);
        ungbr_p(m, Vt, tauw, UngbrOpts{});
    }

    Uplo uplo = (m >= n) ? Uplo::Upper : Uplo::Lower;

    int ierr = svd_qr(uplo, want_u, want_vt, s, e, U, Vt);

    return ierr;
}

}  // namespace tlapack

#endif  // TLAPACK_GESVD_HH
