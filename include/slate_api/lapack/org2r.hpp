/// @file org2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/org2r.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_ORG2R_HH__
#define __SLATE_ORG2R_HH__

#include "lapack/org2r.hpp"

#include "tblas.hpp"

namespace lapack {

/** Generates a m-by-n matrix Q with orthogonal columns.
 * \[
 *     Q  =  H_1 H_2 ... H_k
 * \]
 * 
 * @return  0 if success
 * @return -i if the ith argument is invalid
 * 
 * @param[in] m The number of rows of the matrix A. m>=0
 * @param[in] n The number of columns of the matrix A. n>=0
 * @param[in] k The number of elementary reflectors whose product defines the matrix Q. n>=k>=0
 * @param[in,out] A m-by-n matrix.
 *      On entry, the i-th column must contains the vector which defines the
 *      elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by GEQRF in the
 *      first k columns of its array argument A.
 *      On exit, the m-by-n matrix $Q  =  H_1 H_2 ... H_k$.
 * @param[in] lda The leading dimension of A. lda >= max(1,m).
 * @param[in] tau Real vector of length min(m,n).      
 *      The scalar factors of the elementary reflectors.
 * 
 * @ingroup geqrf
 */
template< typename TA, typename Ttau >
inline int org2r(
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    TA* A, blas::idx_t lda,
    const Ttau* tau )
{
    using blas::internal::colmajor_matrix;
    using blas::internal::vector;

    // check arguments
    lapack_error_if( m < 0, -1 );
    lapack_error_if( n < 0 || n > m, -2 );
    lapack_error_if( k < 0 || k > n, -3 );
    lapack_error_if( lda < m, -5 );

    // quick return
    if (n <= 0) return 0;

    // Local parameters
    int info = 0;
    TA* work = new TA[ (n > 0) ? n-1 : 0 ];

    // Matrix views
    auto _A    = colmajor_matrix<TA>( A, m, n, lda );
    auto _tau  = vector<Ttau>( (Ttau*)tau, std::min<blas::idx_t>( m, n ), 1 );
    auto _work = vector<TA>  ( work, n-1, 1 );
    
    info = org2r( k, _A, _tau, _work );

    delete[] work;
    return info;
}

}

#endif // __SLATE_ORG2R_HH__
