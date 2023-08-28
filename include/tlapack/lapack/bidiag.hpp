/// @file bidiag.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BIDIAG_HH
#define TLAPACK_BIDIAG_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/gebd2.hpp"
#include "tlapack/lapack/gebrd.hpp"

namespace tlapack {

enum class BidiagVariant : char { Level2 = '2', Blocked = 'B' };

struct BidiagOpts : public GebrdOpts {
    BidiagVariant variant = BidiagVariant::Blocked;
};

/** Worspace query of bidiag()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param[in] tauv vector of length min(m,n).
 *
 * @param[in] tauw vector of length min(m,n).
 *
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
constexpr WorkInfo bidiag_worksize(const matrix_t& A,
                                   const vector_t& tauv,
                                   const vector_t& tauw,
                                   const BidiagOpts& opts = {})
{
    // Call variant
    if (opts.variant == BidiagVariant::Level2)
        return gebd2_worksize<T>(A, tauv, tauw);
    else
        return gebrd_worksize<T>(A, tauv, tauw, opts);
}

/** Reduces a general m by n matrix A to an upper
 *  real bidiagonal form B by a unitary transformation:
 * \[
 *          Q**H * A * P = B.
 * \]
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
 *      - @c opts.variant: Variant of the algorithm to use.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int bidiag(matrix_t& A,
           vector_t& tauv,
           vector_t& tauw,
           const BidiagOpts& opts = {})
{
    // Call variant
    if (opts.variant == BidiagVariant::Level2)
        return gebd2(A, tauv, tauw);
    else
        return gebrd(A, tauv, tauw, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_BIDIAG_HH