/// @file orm2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ormr2.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __ORM2R_HH__
#define __ORM2R_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larf.hpp"

namespace lapack {

/** Applies orthogonal matrix Q to a matrix C.
 * 
 * @param work Vector of size n (m if side == Side::Right).
 * @see orm2r( Side, Op, blas::idx_t, blas::idx_t, blas::idx_t, const TA*, blas::idx_t, const blas::real_type<TA,TC>*, TC*, blas::idx_t )
 * 
 * @ingroup geqrf
 */
template<typename TA, typename TC>
int orm2r(
    Side side, Op trans,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    const TA* A, blas::idx_t lda,
    const blas::real_type<TA,TC>* tau,
    TC* C, blas::idx_t ldc,
    blas::scalar_type<TA,TC>* work )
{
    // check arguments

    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );
    lapack_error_if( m < 0, -3 );
    lapack_error_if( n < 0, -4 );

    const idx_t q = (side == Side::Left) ? m : n;
    lapack_error_if( k < 0 || k > q, -5 );
    lapack_error_if( lda < q, -7 );
    lapack_error_if( ldc < m, -10 );

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return 0;

    if( side == Side::Left ) {
        if( trans == Op::NoTrans ) {
            for (idx_t i = 0; i < k; ++i)
                larf( Side::Left, m-k+i+1, n, A+i, lda, tau[i], C, ldc, work );
        }
        else {
            for (idx_t i = k-1; i != idx_t(-1); --i)
                larf( Side::Left, m-k+i+1, n, A+i, lda, tau[i], C, ldc, work );
        }
    }
    else { // side == Side::Right
        if( trans == Op::NoTrans ) {
            for (idx_t i = 0; i < k; ++i)
                larf( Side::Right, m, n-k+i+1, A+i, lda, tau[i], C, ldc, work );
        }
        else {
            for (idx_t i = k-1; i != idx_t(-1); --i)
                larf( Side::Right, m, n-k+i+1, A+i, lda, tau[i], C, ldc, work );
        }
    }

    return 0;
}

}

#endif // __ORM2R_HH__