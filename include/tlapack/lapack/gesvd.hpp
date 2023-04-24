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
template <class idx_t = size_t>
struct gesvd_opts_t : public workspace_opts_t<> {
    inline constexpr gesvd_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};

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
 * @param[in,out]
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup workspace_query
 */
template <class matrix_t, class r_vector_t>
workinfo_t gesvd_worksize(bool want_u,
                          bool want_vt,
                          matrix_t& A,
                          r_vector_t& s,
                          matrix_t& U,
                          matrix_t& Vt,
                          const gesvd_opts_t<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t, r_vector_t>;
    using T = type_t<work_t>;
    using real_t = real_type<T>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);

    workinfo_t myWorkinfo(sizeof(T) * min(m, n), 3);

    workinfo_t extraworkinfo;

    if (m >= n) {
        auto tau = slice(A, 0, pair{0, k});
        gebrd_worksize(A, s, s, tau, tau, extraworkinfo);
        if (want_u) {
            ungbr_q_worksize(n, U, tau, extraworkinfo);
        }
        if (want_vt) {
            ungbr_p_worksize(m, Vt, tau, extraworkinfo);
        }
    }
    else {
        auto tau = slice(A, pair{0, k}, 0);
        gebrd_worksize(A, s, s, tau, tau, extraworkinfo);
        if (want_u) {
            ungbr_q_worksize(n, U, tau, extraworkinfo);
        }
        if (want_vt) {
            ungbr_p_worksize(m, Vt, tau, extraworkinfo);
        }
    }

    myWorkinfo += extraworkinfo;
    return myWorkinfo;
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
template <class matrix_t, class r_vector_t>
int gesvd(bool want_u,
          bool want_vt,
          matrix_t& A,
          r_vector_t& s,
          matrix_t& U,
          matrix_t& Vt,
          const gesvd_opts_t<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using work_t = matrix_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<work_t> new_matrix;

    // constants
    const real_t one(1);
    const type_t<work_t> zero(0);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo =
            gesvd_worksize(want_u, want_vt, A, s, U, Vt, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    Workspace sparework;
    auto workMatrix = new_matrix(work, k, 3, sparework);

    // For now, we use a locally allocated vector, because I don't know how to
    // make a real_t vector wrapper of the correct type.
    std::vector<real_t> e(k);
    // auto e = col(workMatrix, 0);
    auto tauv = col(workMatrix, 1);
    auto tauw = col(workMatrix, 2);

    auto&& gebrdOpts = gebrd_opts_t<>{sparework};
    gebrd(A, s, e, tauv, tauw, gebrdOpts);

    if (want_u) {
        auto ungbrOpts = ungbr_opts_t<matrix_t>{sparework};
        auto Ui = slice(U, pair{0, m}, pair{0, k});
        lacpy(Uplo::Lower, slice(A, pair{0, m}, pair{0, k}), Ui);
        ungbr_q(n, U, tauv, ungbrOpts);
    }

    if (want_vt) {
        auto ungbrOpts = ungbr_opts_t<matrix_t>{sparework};
        auto Vti = slice(Vt, pair{0, k}, pair{0, n});
        lacpy(Uplo::Upper, slice(A, pair{0, k}, pair{0, n}), Vti);
        ungbr_p(m, Vt, tauw, ungbrOpts);
    }

    Uplo uplo = (m >= n) ? Uplo::Upper : Uplo::Lower;

    svd_qr(uplo, want_u, want_vt, s, e, U, Vt);

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GESVD_HH
