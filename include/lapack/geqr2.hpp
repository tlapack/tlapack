/// @file geqr2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/geqr2.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_GEQR2_HH__
#define __TLAPACK_GEQR2_HH__

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack {

/** Computes a QR factorization of a matrix A.
 * 
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 * 
 * @return  0 if success
 * @return -i if the ith argument is invalid
 * 
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *      If
 *          n-1 < m and
 *          type_t<matrix_t> == type_t<vector_t>
 *      then one may use tau[1:n] as the workspace.
 * 
 * @param work Vector of size n-1.
 * 
 * @ingroup geqrf
 */
template< class matrix_t, class vector_t, class work_t >
int geqr2( matrix_t& A, vector_t &tau, work_t &work )
{
    using idx_t = size_type< matrix_t >;
    using pair  = pair<idx_t,idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>( m, n-1 );

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ), -1 );
    tlapack_check_false( (idx_t) size(tau)  < std::min<idx_t>( m, n ), -2 );
    tlapack_check_false( (idx_t) size(work) < n-1, -3 );

    // quick return
    if (n <= 0) return 0;

    for(idx_t i = 0; i < k; ++i) {
      
        // Define v := A[i:m,i]
        auto v = slice( A, pair{i,m}, i );

        // Generate the (i+1)-th elementary Householder reflection on x
        larfg( v, tau[i] );

        // Define v := A[i:m,i] and C := A[i:m,i+1:n], and w := work[i:n-1]
        auto C = slice( A, pair{i,m}, pair{i+1,n} );
        auto w = slice( work, pair{i,n-1} );

        // C := I - tau_i v v^H
        larf( left_side, v, tau[i], C, w );
	}
    if( n-1 < m ) {
        // Define v := A[n-1:m,n-1]
        auto v = slice( A, pair{n-1,m}, n-1 );
        // Generate the n-th elementary Householder reflection on x
        larfg( v, tau[n-1] );
    }

    return 0;
}

} // lapack

#endif // __GEQR2_HH__
