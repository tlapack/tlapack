/// @file ung2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ung2r.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
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

/** Worspace query of ung2r()
 * 
 * @param[in] k
 *      The number of elementary reflectors whose product defines the matrix Q.
 *      Note that: `n >= k >= 0`.
 * 
 * @param[in] A m-by-n matrix.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 * 
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template< class matrix_t, class vector_t >
inline constexpr
void ung2r_worksize(
    size_type< matrix_t > k, const matrix_t& A, const vector_t &tau,
    workinfo_t& workinfo, const workspace_opts_t<>& opts = {} )
{
    using idx_t = size_type< matrix_t >;

    // constants
    const idx_t n = ncols(A);

    if( n > 1 ) {
        auto C = cols( A, range<idx_t>{1,n} );
        larf_worksize( left_side, forward, col(A,0), tau[0], C, workinfo, opts );
    }
}

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
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 * 
 * @return 0 if success 
 * 
 * @ingroup computational
 */
template< class matrix_t, class vector_t >
int ung2r(
    size_type< matrix_t > k, matrix_t& A, const vector_t &tau, const workspace_opts_t<>& opts = {} )
{
    using T      = type_t< matrix_t >;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;
    using pair  = pair<idx_t,idx_t>;
    
    // constants
    const real_t zero( 0 );
    const real_t one ( 1 );
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false( k < 0 || k > n );
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check_false( (idx_t) size(tau)  < k );

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]()
    {
        workinfo_t workinfo;
        ung2r_worksize( k, A, tau, workinfo, opts );
        return alloc_workspace( localworkdata, workinfo, opts.work );
    }();
        
    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{ work };
    
    // Initialise columns k:n-1 to columns of the unit matrix
    for (idx_t j = k; j < n; ++j) {
        for (idx_t l = 0; l < m; ++l)
	        A(l,j) = zero;
        A(j,j) = one;
    }

    for (idx_t i = k-1; i != idx_t(-1); --i) {

        // Apply $H_{i+1}$ to $A( i:m-1, i:n-1 )$ from the left
        if ( i+1 < n ){
            
            // Define v and C
            auto v = slice( A, pair{i,m}, i );
            auto C = slice( A, pair{i,m}, pair{i+1,n} );

            larf( left_side, forward, v, tau[i], C, larfOpts );
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
