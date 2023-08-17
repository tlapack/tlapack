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

/**
 * Options struct for ungbr
 */
struct UngbrOpts {
    size_t nb = 32;  ///< Block size
};

/** Worspace query of ungbr_q()
 *
 * @param[in] k integer.
 *      k is the number of columns
 *      in the original m-by-k matrix reduced by gebrd.
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @param[in] tau vector.
 *      tau is a vector of length min(m,k)
 *      tau(i) must contain the scalar factor of the elementary
 *      reflector H(i), which determines Q, as
 *      returned by gebrd in its array argument tauq.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @param[in] opts Options.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
constexpr WorkInfo ungbr_q_worksize(const size_type<matrix_t> k,
                                    matrix_t& A,
                                    const vector_t& tau,
                                    const UngbrOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (m >= k) {
        return ungqr_worksize<T>(A, tau, opts);
    }
    else {
        auto&& A2 = slice(A, range{0, m - 1}, range{0, m - 1});
        auto&& tau2 = slice(tau, range{0, m - 1});
        return ungqr_worksize<T>(A2, tau2, opts);
    }
}

/** Worspace query of ungbr_p()
 *
 * @param[in] k integer.
 *      k is the number of rows
 *      in the original k-by-n matrix reduced by gebrd.
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @param[in] tau vector.
 *      tau is a vector of length min(k,n)
 *      tau(i) must contain the scalar factor of the elementary
 *      reflector G(i), which determines P**H, as
 *      returned by gebrd in its array argument taup.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @param[in] opts Options.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
constexpr WorkInfo ungbr_p_worksize(const size_type<matrix_t> k,
                                    matrix_t& A,
                                    const vector_t& tau,
                                    const UngbrOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (m >= k) {
        auto&& A2 = slice(A, range{0, n - 1}, range{0, n - 1});
        auto&& tau2 = slice(tau, range{0, n - 1});
        return unglq_worksize<T>(A2, tau2, opts);
    }
    else {
        return unglq_worksize<T>(A, tau, opts);
    }
}

/** Generates the unitary matrix Q
 *  determined by gebrd when reducing a matrix A to bidiagonal
 *  form: A = Q * B * P**H.  Q is defined as a product of
 *  elementary reflectors H(i).
 *
 *  A is assumed to have been an M-by-K matrix, and Q is of order M:
 *  if m >= k, Q = H(1) H(2) . . . H(k) and ungbr returns the first n
 *  columns of Q, where m >= n >= k;
 *  if m < k, Q = H(1) H(2) . . . H(m-1) and ungbr returns Q as an
 *  M-by-M matrix.
 *
 * @param[in] k integer.
 *      k is the number of columns
 *      in the original m-by-k matrix reduced by gebrd.
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the vectors which define the elementary reflectors
 *      as returned by gebrd.
 *      On exit, the m-by-n matrix Q.
 *
 * @param[in] tau vector.
 *      tau is a vector of length min(m,k)
 *      tau(i) must contain the scalar factor of the elementary
 *      reflector H(i), which determines Q, as
 *      returned by gebrd in its array argument tauq.
 *
 * @param[in] opts Options.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int ungbr_q(const size_type<matrix_t> k,
            matrix_t& A,
            const vector_t& tau,
            const UngbrOpts& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t m = nrows(A);

    UngqrOpts ungqrOpts;
    ungqrOpts.nb = opts.nb;
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
            auto A2 = slice(A, range{1, m}, range{1, m});
            auto tau2 = slice(tau, range{0, m - 1});
            ungqr(A2, tau2, ungqrOpts);
        }
    }

    return 0;
}

/** Generates the unitary matrix P**H
 *  determined by gebrd when reducing a matrix A to bidiagonal
 *  form: A = Q * B * P**H.  P**H is defined as a product of
 *  elementary reflectors G(i).
 *
 *  A is assumed to have been an K-by-N matrix, and P**H
 *  is of order N:
 *  if k < n, P**H = G(k) . . . G(2) G(1) and ungbr_p returns the first m
 *  rows of P**H, where n >= m >= k;
 *  if k >= n, P**H = G(n-1) . . . G(2) G(1) and ungbr_p returns P**H as an
 *  N-by-N matrix.
 *
 * @param[in] k integer.
 *      k is the number of rows
 *      in the original k-by-n matrix reduced by gebrd.
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the vectors which define the elementary reflectors
 *      as returned by gebrd. On exit, the m-by-n matrix P**H.
 *
 * @param[in] tau vector.
 *      tau is a vector of length min(k,n)
 *      tau(i) must contain the scalar factor of the elementary
 *      reflector G(i), which determines P**H, as
 *      returned by gebrd in its array argument taup.
 *
 * @param[in] opts Options.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int ungbr_p(const size_type<matrix_t> k,
            matrix_t& A,
            const vector_t& tau,
            const UngbrOpts& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const idx_t n = ncols(A);

    UnglqOpts unglqOpts;
    unglqOpts.nb = opts.nb;
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
            auto A2 = slice(A, range{1, n}, range{1, n});
            auto tau2 = slice(tau, range{0, n - 1});
            unglq(A2, tau2, unglqOpts);
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_UNGBR_HH
