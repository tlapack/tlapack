/// @file ung2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ung2r.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNG2R_HH
#define TLAPACK_UNG2R_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/blas/scal.hpp"

namespace tlapack {

/**
 * @brief Generates a matrix Q with orthogonal columns.
 * \[
 *     Q  =  H_1 H_2 ... H_k
 * \]
 * 
 * @param[in] k
 *      The number of elementary reflectors whose product defines the matrix Q.
 *      Note that: `n >= k >= 0`.
 * 
 * @param[in,out] A m-by-n matrix.
 *      On entry, the i-th column must contains the vector which defines the
 *      elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by geqrf.
 *      On exit, the m-by-n matrix $Q$.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 * 
 * @param work Vector of at least size n-1.
 * 
 * @return 0 if success 
 * 
 * @ingroup geqrf
 */
template< class matrix_t, class vector_t, class work_t >
int ung2r(
    size_type< matrix_t > k, matrix_t& A, const vector_t &tau, work_t &work )
{
    using T      = type_t< matrix_t >;
    using idx_t  = size_type< matrix_t >;
    using pair  = pair<idx_t,idx_t>;
    
    // constants
    const T zero( 0.0 );
    const T one ( 1.0 );
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false( k < 0 || k > n );
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check_false( (idx_t) size(tau)  < k );
    tlapack_check_false( (idx_t) size(work) < n-1 );

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

#endif // TLAPACK_UNG2R_HH
