/// @file qr_iteration.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_QR_ITERATION_HH
#define TLAPACK_QR_ITERATION_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lahqr.hpp"
#include "tlapack/lapack/multishift_qr.hpp"

namespace tlapack {

enum class QRIterationVariant : char { MultiShift = 'M', DoubleShift = 'D' };

struct QRIterationOpts : public FrancisOpts {
    QRIterationVariant variant = QRIterationVariant::MultiShift;
};

/** Worspace query of qr_iteration()
 *
 * @param[in] want_t bool.
 *      If true, the full Schur factor T will be computed.
 * @param[in] want_z bool.
 *      If true, the Schur vectors Z will be computed.
 * @param[in] ilo    integer.
 *      Either ilo=0 or A(ilo,ilo-1) = 0.
 * @param[in] ihi    integer.
 *      The matrix A is assumed to be already quasi-triangular in rows and
 *      columns ihi:n.
 * @param[in] A  n by n matrix.
 * @param[in] w  n vector.
 * @param[in] Z  n by n matrix.
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
constexpr WorkInfo qr_iteration_worksize(bool want_t,
                                         bool want_z,
                                         size_type<matrix_t> ilo,
                                         size_type<matrix_t> ihi,
                                         const matrix_t& A,
                                         const vector_t& w,
                                         const matrix_t& Z,
                                         const QRIterationOpts& opts = {})
{
    // Call variant
    if (opts.variant == QRIterationVariant::MultiShift)
        return multishift_qr_worksize<T>(want_t, want_z, ilo, ihi, A, w, Z,
                                         opts);
    else
        return WorkInfo(0);
}

/**
 * @brief Computes the eigenvalues and optionally the Schur
 *  factorization of an upper Hessenberg matrix.
 *
 *
 * @return  0 if success
 * @return -i if the ith argument is invalid
 * @return  i if the QR algorithm failed to compute all the eigenvalues
 *            in a total of 30 iterations per eigenvalue. elements
 *            i:ihi of w contain those eigenvalues which have been
 *            successfully computed.
 *
 * @param[in] want_t bool.
 *      If true, the full Schur factor T will be computed.
 * @param[in] want_z bool.
 *      If true, the Schur vectors Z will be computed.
 * @param[in] ilo    integer.
 *      Either ilo=0 or A(ilo,ilo-1) = 0.
 * @param[in] ihi    integer.
 *      The matrix A is assumed to be already quasi-triangular in rows and
 *      columns ihi:n.
 * @param[in,out] A  n by n matrix.
 *      On entry, the matrix A.
 *      On exit, if info=0 and want_t=true, the Schur factor T.
 *      T is quasi-triangular in rows and columns ilo:ihi, with
 *      the diagonal (block) entries in standard form (see above).
 * @param[out] w  size n vector.
 *      On exit, if info=0, w(ilo:ihi) contains the eigenvalues
 *      of A(ilo:ihi,ilo:ihi). The eigenvalues appear in the same
 *      order as the diagonal (block) entries of T.
 * @param[in,out] Z  n by n matrix.
 *      On entry, the previously calculated Schur factors
 *      On exit, the orthogonal updates applied to A are accumulated
 *      into Z.
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
int qr_iteration(bool want_t,
                 bool want_z,
                 size_type<matrix_t> ilo,
                 size_type<matrix_t> ihi,
                 matrix_t& A,
                 vector_t& w,
                 matrix_t& Z,
                 QRIterationOpts& opts)
{
    // Call variant
    if (opts.variant == QRIterationVariant::MultiShift)
        return multishift_qr(want_t, want_z, ilo, ihi, A, w, Z, opts);
    else
        return lahqr(want_t, want_z, ilo, ihi, A, w, Z);
}

template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
int qr_iteration(bool want_t,
                 bool want_z,
                 size_type<matrix_t> ilo,
                 size_type<matrix_t> ihi,
                 matrix_t& A,
                 vector_t& w,
                 matrix_t& Z)
{
    // Call variant
    if (QRIterationOpts().variant == QRIterationVariant::MultiShift)
        return multishift_qr(want_t, want_z, ilo, ihi, A, w, Z);
    else
        return lahqr(want_t, want_z, ilo, ihi, A, w, Z);
}

}  // namespace tlapack

#endif