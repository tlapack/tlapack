/// @file ungbr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zungbr.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGBR_HH
#define TLAPACK_UNGBR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/unglq.hpp"
#include "tlapack/lapack/ungqr.hpp"

namespace tlapack {

enum class QorP : char { Q = 'Q', P = 'P' };

/**
 * Options struct for ungbr
 */
template <class workT_t = void>
struct ungbr_opts_t : public workspace_opts_t<workT_t> {
    inline constexpr ungbr_opts_t(const workspace_opts_t<workT_t>& opts = {})
        : workspace_opts_t<workT_t>(opts){};

    size_type<workT_t> nb = 32;  ///< Block size
};

/** Worspace query of ungbr()
 *
 * @param[in] qp Determines which of the matrices to be generated.
 *
 * @param[in] k integer.
 *      If qp == QorP::Q : k is the number of columns
 *      in the original m-by-k matrix reduced by gebrd.
 *      If qp == QorP::P : k is the number of rows
 *      in the original k-by-n matrix reduced by gebrd.
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @param[in] tau vector.
 *      If qp == QorP::Q : tau is a vector of length min(m,k)
 *      If qp == QorP::P : tau is a vector of length min(k,n)
 *      tau(i) must contain the scalar factor of the elementary
 *      reflector H(i) or G(i), which determines Q or P**H, as
 *      returned by gebrd in its array argument tauq or taup.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @param[in] opts Options.
 *
 * @ingroup workspace_query
 */
template <class matrix_t, class vector_t, class workT_t = void>
inline constexpr void ungbr_worksize(QorP qp,
                                     const size_type<matrix_t> k,
                                     matrix_t& A,
                                     const vector_t& tau,
                                     workinfo_t& workinfo,
                                     const ungbr_opts_t<workT_t>& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using pair = std::pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (qp == QorP::Q) {
        if (m >= k) {
            ungqr_worksize(A, tau, workinfo, opts);
        }
        else {
            auto A2 = slice(A, pair{0, m - 1}, pair{0, m - 1});
            auto tau2 = slice(tau, pair{0, m - 1});
            ungqr_worksize(A2, tau2, workinfo, opts);
        }
    }
    else {
        if (m >= k) {
            auto A2 = slice(A, pair{0, n - 1}, pair{0, n - 1});
            auto tau2 = slice(tau, pair{0, n - 1});
            unglq_worksize(A2, tau2, workinfo, opts);
        }
        else {
            unglq_worksize(A, tau, workinfo, opts);
        }
    }
}

/** Generates one of the unitary matrices Q or P**H
 *  determined by gebrd when reducing a matrix A to bidiagonal
 *  form: A = Q * B * P**H.  Q and P**H are defined as products of
 *  elementary reflectors H(i) or G(i) respectively.
 *
 *  If qp = Q, A is assumed to have been an M-by-K matrix, and Q
 *  is of order M:
 *  if m >= k, Q = H(1) H(2) . . . H(k) and ungbr returns the first n
 *  columns of Q, where m >= n >= k;
 *  if m < k, Q = H(1) H(2) . . . H(m-1) and ungbr returns Q as an
 *  M-by-M matrix.
 *
 * @param[in] qp Determines which of the matrices to be generated.
 *
 * @param[in] k integer.
 *      If qp == QorP::Q : k is the number of columns
 *      in the original m-by-k matrix reduced by gebrd.
 *      If qp == QorP::P : k is the number of rows
 *      in the original k-by-n matrix reduced by gebrd.
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the vectors which define the elementary reflectors
 *      as returned by gebrd.
 *      On exit, the m-by-n matrix Q or P**H.
 *
 * @param[in] tau vector.
 *      If qp == QorP::Q : tau is a vector of length min(m,k)
 *      If qp == QorP::P : tau is a vector of length min(k,n)
 *      tau(i) must contain the scalar factor of the elementary
 *      reflector H(i) or G(i), which determines Q or P**H, as
 *      returned by gebrd in its array argument tauq or taup.
 *
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <class matrix_t, class vector_t, class workT_t = void>
int ungbr(QorP qp,
          const size_type<matrix_t> k,
          matrix_t& A,
          const vector_t& tau,
          const ungbr_opts_t<workT_t>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (qp == QorP::Q) {
        ungqr_opts_t<matrix_t> ungqrOpts;
        ungqrOpts.nb = opts.nb;
        ungqrOpts.work = opts.work;
        //
        // Form Q, determined by a call to gebrd to reduce an m-by-k
        // matrix
        //
        if (m >= k) {
            // If m >= k, assume m >= n >= k
            ungqr(A, tau, ungqrOpts);
        }
        else {
            // Shift the vectors which define the elementary reflectors one
            // column to the right, and set the first row and column of Q
            // to those of the unit matrix
            for (idx_t j = m - 1; j > 0; --j) {
                A(0, j) = zero;
                for (idx_t i = j + 1; i < m; ++i)
                    A(i, j) = A(i, j - 1);
            }
            A(0, 0) = one;
            for (idx_t i = 1; i < m; ++i)
                A(i, 0) = zero;
            if (m > 1) {
                // Form Q(1:m,1:m)
                auto A2 = slice(A, pair{1, m}, pair{1, m});
                auto tau2 = slice(tau, pair{0, m - 1});
                ungqr(A2, tau2, ungqrOpts);
            }
        }
    }
    else {
        unglq_opts_t<matrix_t> unglqOpts;
        unglqOpts.nb = opts.nb;
        unglqOpts.work = opts.work;
        //
        // Form P**H, determined by a call to gebrd to reduce a k-by-n
        // matrix
        //
        if (k < n) {
            // If m >= k, assume m >= n >= k
            unglq(A, tau, unglqOpts);
        }
        else {
            // Shift the vectors which define the elementary reflectors one
            // row downward, and set the first row and column of P**H to
            // those of the unit matrix
            A(0, 0) = one;
            for (idx_t i = 1; i < n; ++i) {
                A(i, 0) = zero;
            }
            for (idx_t j = 1; j < n; ++j) {
                for (idx_t i = j - 1; i > 0; --i) {
                    A(i, j) = A(i - 1, j);
                }
                A(0, j) = zero;
            }
            if (n > 1) {
                auto A2 = slice(A, pair{1, n}, pair{1, n});
                auto tau2 = slice(tau, pair{0, n - 1});
                unglq(A2, tau2, unglqOpts);
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNGBR_HH
