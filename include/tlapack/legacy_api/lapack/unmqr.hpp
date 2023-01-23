/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from geqrf()
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_UNMQR_HH
#define TLAPACK_LEGACY_UNMQR_HH

#include "tlapack/lapack/unmqr.hpp"

namespace tlapack {

/** Multiplies the general m-by-n matrix C by Q from geqrf() using a blocked
 code as follows:
 *
 * @param[in] side
 *     - Side::Left:  apply $Q$ or $Q^H$ from the Left;
 *     - Side::Right: apply $Q$ or $Q^H$ from the Right.
 *
 * @param[in] trans
 *     - Op::NoTrans:   No transpose, apply $Q$;
 *     - Op::ConjTrans: Conjugate transpose, apply $Q^H$.
 *
 * @param[in] m
 *     The number of rows of the matrix C. m >= 0.
 *
 * @param[in] n
 *     The number of columns of the matrix C. n >= 0.
 *
 * @param[in] k
 *     The number of elementary reflectors whose product defines
 *     the matrix Q.
 *     - If side = Left,  m >= k >= 0;
 *     - if side = Right, n >= k >= 0.
 *
 * @param[in] A
 *     - If side = Left,  the m-by-k matrix A, stored in an lda-by-k array;
 *     - if side = Right, the n-by-k matrix A, stored in an lda-by-k array.
 *     \n
 *     The i-th column must contain the vector which defines the
 *     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
 *     geqrf() in the first k columns of its array argument A.
 *
 * @param[in] lda
 *     The leading dimension of the array A.
 *     - If side = Left,  lda >= max(1,m);
 *     - if side = Right, lda >= max(1,n).
 *
 * @param[in] tau
 *     The vector tau of length k.
 *     tau[i] must contain the scalar factor of the elementary
 *     reflector H(i), as returned by geqrf().
 *
 * @param[in,out] C
 *     The m-by-n matrix C, stored in an ldc-by-n array.
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by
 *     $Q C$ or $Q^H C$ or $C Q^H$ or $C Q$.
 *
 * @param[in] ldc
 *     The leading dimension of the array C. ldc >= max(1,m).
 *
 * @see unmqr(
    side_t side, trans_t trans,
    const matrixA_t& A, const tau_t& tau,
    matrixC_t& C, const unmqr_opts_t<workT_t>& opts = {} )
 *
 * @ingroup legacy_lapack
 */
template <class side_t, class trans_t, typename TA, typename TC>
inline int unmqr(side_t side,
                 trans_t trans,
                 idx_t m,
                 idx_t n,
                 idx_t k,
                 TA const* A,
                 idx_t lda,
                 TA const* tau,
                 TC* C,
                 idx_t ldc)
{
    using internal::colmajor_matrix;
    using internal::vector;

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans);

    // Matrix views
    const auto A_ = (side == Side::Left)
                        ? colmajor_matrix<TA>((TA*)A, m, k, lda)
                        : colmajor_matrix<TA>((TA*)A, n, k, lda);
    const auto tau_ = vector((TA*)tau, k);
    auto C_ = colmajor_matrix<TC>(C, m, n, ldc);

    return unmqr(side, trans, A_, tau_, C_);
}

}  // namespace tlapack

#endif  // TLAPACK_LEGACY_UNMQR_HH
