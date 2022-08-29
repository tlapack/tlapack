/// @file geqr2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/geqr2.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQR2_HH
#define TLAPACK_GEQR2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larf.hpp"

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
template< class matrix_t, class vector_t,
    // opts_t:
    class T = scalar_type<
        type_t< matrix_t >,
        type_t< vector_t >
    >,
    class idx_t = size_type< matrix_t >,
    class work_t = legacyVector<T,idx_t>
>
int geqr2( matrix_t& A, vector_t &tau,
    const workspace_opts_t<T,idx_t,work_t>& opts = {} )
{
    using pair  = pair<idx_t,idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>( m, n-1 );

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check_false( (idx_t) size(tau)  < std::min<idx_t>( m, n ) );

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localwork; size_t lwork;
    geqr2_worksize( A, tau, opts );
    byte* work = alloc_workspace( localwork, lwork, opts.work, opts.lwork );

    for(idx_t i = 0; i < k; ++i) {
      
        // Define v := A[i:m,i]
        auto v = slice( A, pair{i,m}, i );

        // Generate the (i+1)-th elementary Householder reflection on x
        larfg( v, tau[i] );

        // Define v := A[i:m,i] and C := A[i:m,i+1:n], and w := work[i:n-1]
        auto C = slice( A, pair{i,m}, pair{i+1,n} );
        
        // C := I - conj(tau_i) v v^H
        opts.work
        larf( left_side, v, conj(tau[i]), C, opts );
	}
    if( n-1 < m ) {
        // Define v := A[n-1:m,n-1]
        auto v = slice( A, pair{n-1,m}, n-1 );
        // Generate the n-th elementary Householder reflection on x
        larfg( v, tau[n-1] );
    }

    return 0;
}

template< class matrix_t, class vector_t,
    // opts_t:
    class T = scalar_type<
        type_t< matrix_t >,
        type_t< vector_t >
    >,
    class idx_t = size_type< matrix_t >,
    class work_t = legacyVector<T,idx_t>
>
inline constexpr
void geqr2_worksize( matrix_t& A, vector_t &tau, size_t& worksize,
    const workspace_opts_t<T,idx_t,work_t>& opts = {} )
{
    larf_worksize( left_side, col(A,0), tau[0], A, worksize, opts );
}

} // lapack

#endif // TLAPACK_GEQR2_HH
