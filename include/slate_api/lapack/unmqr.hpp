/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from `lapack::geqrf`
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_UNMQR_HH__
#define __SLATE_UNMQR_HH__

#include "lapack/unmqr.hpp"

namespace lapack {

/** Multiplies the general m-by-n matrix C by Q from `lapack::geqrf` using a blocked code as follows:
 *
 * - side = Left,  trans = NoTrans:   $Q C$
 * - side = Right, trans = NoTrans:   $C Q$
 * - side = Left,  trans = ConjTrans: $Q^H C$
 * - side = Right, trans = ConjTrans: $C Q^H$
 *
 * where Q is a unitary matrix defined as the product of k
 * elementary reflectors, as returned by `lapack::geqrf`:
 * \[
 *     Q = H(1) H(2) \dots H(k).
 * \]
 *
 * Q is of order m if side = Left and of order n if side = Right.
 *
 * For real matrices, this is an alias for `lapack::ormqr`.
 * 
 * @return  0 if success
 * @return -i if the ith argument is invalid
 *
 * @param[in] side
 *     - lapack::Side::Left:  apply $Q$ or $Q^H$ from the Left;
 *     - lapack::Side::Right: apply $Q$ or $Q^H$ from the Right.
 *
 * @param[in] trans
 *     - lapack::Op::NoTrans:   No transpose, apply $Q$;
 *     - lapack::Op::ConjTrans: Conjugate transpose, apply $Q^H$.
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
 *     `lapack::geqrf` in the first k columns of its array argument A.
 *
 * @param[in] lda
 *     The leading dimension of the array A.
 *     - If side = Left,  lda >= max(1,m);
 *     - if side = Right, lda >= max(1,n).
 *
 * @param[in] tau
 *     The vector tau of length k.
 *     tau[i] must contain the scalar factor of the elementary
 *     reflector H(i), as returned by `lapack::geqrf`.
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
 * @ingroup geqrf
 */
template< typename TA, typename TC >
inline int unmqr(
    Side side, Op trans,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    TA const* A, blas::idx_t lda,
    TA const* tau,
    TC* C, blas::idx_t ldc )
{
    typedef blas::scalar_type<TA,TC> scalar_t;
    using blas::internal::colmajor_matrix;
    using blas::internal::vector;

    // Constants
    const int nb = 32;      // number of blocks
    
    // Allocate work
    const idx_t nw = (side == Side::Left)
                ? ( (n >= 1) ? n : 1 )
                : ( (m >= 1) ? m : 1 );
    scalar_t* work = new scalar_t[
        nw*nb + nb*nb
    ];

    // Matrix views
    const auto _A = (side == Side::Left)
            ? colmajor_matrix<TA>( (TA*)A, m, k, lda )
            : colmajor_matrix<TA>( (TA*)A, n, k, lda );
    const auto _tau = vector<TA>( (TA*)tau, k, 1 );
    auto _C = colmajor_matrix<TC>( C, m, n, ldc );
    auto _W = colmajor_matrix<scalar_t>( work, nb, nw+nb );
    
    int info = 0;
    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            info = unmqr(
                left_side, noTranspose,
                _A, _tau, _C, _W );
        } else if (trans == Op::Trans) {
            info = unmqr(
                left_side, transpose,
                _A, _tau, _C, _W );
        } else { // (trans == Op::ConjTrans)
            info = unmqr(
                left_side, conjTranspose,
                _A, _tau, _C, _W );
        }
    }
    else { // side == Side::Right
        if (trans == Op::NoTrans) {
            info = unmqr(
                right_side, noTranspose,
                _A, _tau, _C, _W );
        } else if (trans == Op::Trans) {
            info = unmqr(
                right_side, transpose,
                _A, _tau, _C, _W );
        } else { // (trans == Op::ConjTrans)
            info = unmqr(
                right_side, conjTranspose,
                _A, _tau, _C, _W );
        }
    }

    delete[] work;
    return info;
}

}

#endif // __SLATE_UNMQR_HH__
