/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from `lapack::geqrf`
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __UNMQR_HH__
#define __UNMQR_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larft.hpp"

namespace lapack {

/** Multiplies the general m-by-n matrix C by Q from `lapack::geqrf` using a blocked code.
 * 
 * @param work Vector of size n*nb + nb*nb (m*nb + nb*nb if side == Side::Right).
 * @see unmqr( Side, Op, blas::idx_t, blas::idx_t, blas::idx_t, const TA*, blas::idx_t, const blas::real_type<TA,TC>*, TC*, blas::idx_t )
 * 
 * @ingroup geqrf
 */
template< typename TA, typename TC >
int unmqr(
    Side side, Op trans,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    TA const* A, blas::idx_t lda,
    TA const* tau,
    TC* C, blas::idx_t ldc,
    blas::scalar_type<TA,TC>* work )
{
    using blas::max;
    using blas::min;

    // Constants
    const int nb = 32;      // number of blocks

    // Choose side 
    idx_t nQ, nw;
    if( side == Side::Left ){ nQ = m; nw = max( 1, n ); }
    else                    { nQ = n; nw = max( 1, m ); }

    // check arguments
    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );
    lapack_error_if( m < 0, -3 );
    lapack_error_if( n < 0, -4 );
    lapack_error_if( k < 0 || k > nQ, -5 );
    lapack_error_if( lda < nQ, -7 );
    lapack_error_if( ldc < m, -10 );

    // Quick return
    if (m == 0 || n == 0 || k == 0)
        return 0;

    #define _A(i_, j_) A[ (i_) + (j_)*lda ]
    #define _C(i_, j_) C[ (i_) + (j_)*ldc ]

    // Preparing loop indexes
    idx_t i0, iN, step;
    if( (side == Side::Left && trans != Op::NoTrans) ||
        (side != Side::Left && trans == Op::NoTrans) ) {
        i0 = 0;
        iN = k-1+nb;
        step = nb;
    }
    else {
        i0 = ( (k-1) / nb ) * nb;
        iN = -nb;
        step = -nb;
    }
    idx_t mi, ic, ni, jc;
    if( side == Side::Left ) { ni = n-1; jc = 0; }
    else                     { mi = m-1; ic = 0; }
    
    // Main loop
    for (idx_t i = i0; i != iN; i += step) {
        idx_t ib = min( nb, k-i );

        // Form the triangular factor of the block reflector
        // $H = H(i) H(i+1) ... H(i+ib-1)$
        lapack::larft(  Direction::Forward, StoreV::Columnwise,
                        nQ-i, ib, &_A(i,i), lda, &tau[i], &work[nw*nb], nb );

        // H or H**H is applied to C[0:m-1,0:n-1]
        if( side == Side::Left ) { mi = m-i; ic = i; }
        else                     { ni = n-i; jc = i; }

        // Apply H or H**H
        lapack::larfb(  side, trans, Direction::Forward, StoreV::Columnwise,
                        mi, ni, ib, &_A(i,i), lda, &work[nw*nb], nb,
                        &_C(ic,jc), ldc, work );
    }

    return 0;

    #undef _A
    #undef _C
}

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
 *     tau(i) must contain the scalar factor of the elementary
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

    // Constants
    const int nb = 32;      // number of blocks
    
    // Allocate work
    const idx_t nw = (side == Side::Left)
                ? ( (n >= 1) ? n : 1 )
                : ( (m >= 1) ? m : 1 );
    scalar_t* work = new scalar_t[
        nw*nb + nb*nb
    ];
    
    // main call
    int info = unmqr( side, trans, m, n, k, A, lda, tau, C, ldc, work );

    delete[] work;
    return info;
}

}

#endif // __UNMQR_HH__
