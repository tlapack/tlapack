/// @file unm2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ormr2.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_UNM2R_HH__
#define __TLAPACK_LEGACY_UNM2R_HH__

#include "lapack/unm2r.hpp"

namespace tlapack {

/** Applies unitary matrix Q to a matrix C.
 * 
 * @return  0 if success
 * @return -i if the ith argument is invalid
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
template< class side_t, class trans_t, typename TA, typename TC>
inline int unm2r(
    side_t side, trans_t trans,
    idx_t m, idx_t n, idx_t k,
    TA* A, idx_t lda,
    const real_type<TA,TC>* tau,
    TC* C, idx_t ldc )
{
    typedef scalar_type<TA,TC> scalar_t;
    using internal::colmajor_matrix;
    using internal::vector;

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

    scalar_t* work = new scalar_t[ (q > 0) ? q : 0 ];

    // Matrix views
    const auto A_ = colmajor_matrix<TA>( (TA*)A, q, k, lda );
    const auto _tau = vector( (TA*)tau, k );
    auto C_ = colmajor_matrix<TC>( C, m, n, ldc );
    auto _work = vector( work, q );

    int info = unm2r( side, trans, A, _tau, C_, _work );

    delete[] work;
    return info;
}

}

#endif // __TLAPACK_LEGACY_UNM2R_HH__