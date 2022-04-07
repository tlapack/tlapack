/// @file ung2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ung2r.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __UNG2R_HH__
#define __UNG2R_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larf.hpp"

#include "tblas.hpp"

namespace lapack {

/** Generates a m-by-n matrix Q with orthogonal columns.
 * 
 * @param work Vector of size n-1.
 *     It is possible to use the subarray tau[1:n-1] as the work vector, i.e.,
 *         ung2r( ..., tau, &(tau[1]) )
 *     and, in this case, the original vector tau is lost. 
 * @see ung2r( blas::idx_t, blas::idx_t, blas::idx_t, TA*, blas::idx_t, const Ttau* )
 * 
 * @ingroup geqrf
 */
template< class matrix_t, class vector_t, class work_t >
int ung2r(
    size_type< matrix_t > k, matrix_t& A, vector_t &tau, work_t &work )
{
    using blas::scal;
    using T      = type_t< matrix_t >;
    using idx_t  = size_type< matrix_t >;
    using pair  = std::pair<idx_t,idx_t>;
    
    // constants
    const T zero( 0.0 );
    const T one ( 1.0 );
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    lapack_error_if( access_denied( dense, write_policy(A) ), -2 );
    lapack_error_if( (idx_t) size(tau)  < std::min<idx_t>( m, n ), -3 );
    lapack_error_if( (idx_t) size(work) < n-1, -4 );

    // quick return
    if (n <= 0) return 0;
    
    // Initialise columns k:n-1 to columns of the unit matrix
    for (idx_t j = k; j < n; ++j) {
        for (idx_t l = 0; l < m; ++l)
	        A(l,j) = zero;
        A(j,j) = one;
    }

    for (idx_t i = k-1; i != idx_t(-1); --i) {

        // Apply $H_{i+1}$ to $A( i:m-1, i:n-1 )$ from the left
        if ( i+1 < n ){
            A(i,i) = one;

            // Define v and C
            auto v = slice( A, pair{i,m}, i );
            auto C = slice( A, pair{i,m}, pair{i+1,n} );
            auto w = slice( work, pair{i,n-1} );

            larf( left_side, std::move(v), tau[i], C, w );
        }
        if ( i+1 < m ) {
            auto v = slice( A, pair{i+1,m}, i );
            scal( -tau[i], v );
        }
        A(i,i) = one - tau[i];

        // Set A( 0:i-1, i ) to zero
        for (idx_t l = 0; l < i; l++)
            A(l,i) = zero;
    }

    return 0;
}

}

#endif // __UNG2R_HH__
