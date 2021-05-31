// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Adapted from https://github.com/langou/latl/blob/master/include/ormr2.h
/// @author Rodney James, University of Colorado Denver, USA

#ifndef __ORMR2_HH__
#define __ORMR2_HH__

#include "blas/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larf.hpp"

namespace lapack {

/** Applies orthogonal matrix Q to a matrix C.
 * 
 * @param work Vector of size n (m if side == Side::Right).
 * @see ormr2( Side, Op, blas::size_t, blas::size_t, blas::size_t, const TA*, blas::size_t, const blas::real_type<TA,TC>*, TC*, blas::size_t )
 * 
 * @ingroup geqrf
 */
template<typename TA, typename TC>
void ormr2(
    Side side, Op trans,
    blas::size_t m, blas::size_t n, blas::size_t k,
    const TA* A, blas::size_t lda,
    const blas::real_type<TA,TC>* tau,
    TC* C, blas::size_t ldc,
    blas::scalar_type<TA,TC>* work )
{
    // check arguments

    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    const size_t q = (side == Side::Left) ? m : n;
    blas_error_if( k < 0 || k > q );
    blas_error_if( lda < q );
    blas_error_if( ldc < m );

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return;

    if( side == Side::Left ) {
        if( trans == Op::NoTrans ) {
            for (size_t i = 0; i < k; ++i)
                larf( Side::Left, m-k+i+1, n, A+i, lda, tau[i], C, ldc, work );
        }
        else {
            for (size_t i = k-1; i > size_t(-1); --i)
                larf( Side::Left, m-k+i+1, n, A+i, lda, tau[i], C, ldc, work );
        }
    }
    else { // side == Side::Right
        if( trans == Op::NoTrans ) {
            for (size_t i = 0; i < k; ++i)
                larf( Side::Right, m, n-k+i+1, A+i, lda, tau[i], C, ldc, work );
        }
        else {
            for (size_t i = k-1; i > size_t(-1); --i)
                larf( Side::Right, m, n-k+i+1, A+i, lda, tau[i], C, ldc, work );
        }
    }
}

/** Applies orthogonal matrix Q to a matrix C.
 * 
 * @param[in] side Specifies which side Q is to be applied.
 *                 'L': apply Q or Q' from the Left;
 *                 'R': apply Q or Q' from the Right.
 * @param[in] trans Specifies whether Q or Q' is applied.
 *                 'N':  No transpose, apply Q;
 *                 'T':  Transpose, apply Q'.
 * @param[in] m The number of rows of the matrix C.
 * @param[in] n The number of columns of the matrix C.
 * @param[in] k The number of elementary reflectors whose product defines the matrix Q.
 *                 If side='L', m>=k>=0;
 *                 if side='R', n>=k>=0.
 * @param[in] A Matrix containing the elementary reflectors H.
 *                 If side='L', A is k-by-m;
 *                 if side='R', A is k-by-n.
 * @param[in] ldA The column length of the matrix A.  ldA>=k.
 * @param[in] tau Real vector of length k containing the scalar factors of the
 * elementary reflectors.
 * @param[in,out] C m-by-n matrix. 
 *     On exit, C is replaced by one of the following:
 *                 If side='L' & trans='N':  C <- Q * C
 *                 If side='L' & trans='T':  C <- Q'* C
 *                 If side='R' & trans='T':  C <- C * Q'
 *                 If side='R' & trans='N':  C <- C * Q
 * @param ldC The column length the matrix C. ldC>=m.
 * 
 * @ingroup geqrf
 */
template<typename TA, typename TC>
inline void ormr2(
    Side side, Op trans,
    blas::size_t m, blas::size_t n, blas::size_t k,
    const TA* A, blas::size_t lda,
    const blas::real_type<TA,TC>* tau,
    TC* C, blas::size_t ldc )
{
    typedef blas::scalar_type<TA,TC> scalar_t;

    scalar_t* work = new scalar_t[
        (side == Side::Left)
            ? ( (m >= 0) ? m : 0 )
            : ( (n >= 0) ? n : 0 )
    ];
    ormr2( side, trans, m, n, k, A, lda, tau, C, ldc, work );
    delete[] work;
}

}

#endif // __ORMR2_HH__