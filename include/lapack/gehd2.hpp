/// @file gehd2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dgehd2.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_GEHD2_HH__
#define __TLAPACK_GEHD2_HH__

#include <iostream>

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack {

/** Reduces a general square matrix to upper Hessenberg form
 * 
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_ilo H_ilo+1 ... H_ihi,
 * \]
 * Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i] = 0; v[i+1] = 1,
 * \]
 * with v[i+2] through v[ihi] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 * 
 * @return  0 if success
 * @return -i if the ith argument is invalid
 * 
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      It is assumed that A is already upper Hessenberg in columns
 *      0:ilo and rows ihi:n and is already upper triangular in
 *      columns ihi+1:n and rows 0:ilo.
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in,out] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 *      On exit, the upper triangle and the first subdiagonal of A
 *      are overwritten with the upper Hessenberg matrix H, and the
 *      elements below the first subdiagonal, with the array TAU,
 *      represent the orthogonal matrix Q as a product of elementary
 *      reflectors. See Further Details.
 * @param[out] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 * @param work Vector of size n.
 * 
 * @ingroup gehrd
 */
template< class matrix_t, class vector_t, class work_t >
int gehd2( size_type< matrix_t > ilo, size_type< matrix_t > ihi, matrix_t& A, vector_t &tau, work_t &work )
{
    using TA    = type_t< matrix_t >;
    using idx_t = size_type< matrix_t >;
    using pair  = pair<idx_t,idx_t>;

    // constants
    const TA one( 1 );
    const idx_t n = ncols(A);

    // check arguments
    lapack_error_if( access_denied( dense, write_policy(A) ), -3 );
    lapack_error_if( ncols(A) != nrows(A), -3 );
    lapack_error_if( (idx_t) size(tau)  < n-1, -4 );
    lapack_error_if( (idx_t) size(work) < n, -5 );

    // quick return
    if (n <= 0) return 0;

    for(idx_t i = ilo; i < ihi-1; ++i) {

        // Define x := A[i+1:ihi,i]
        auto v = slice(A, pair{i+1,ihi}, i);

        // Generate the (i+1)-th elementary Householder reflection on x
        larfg( v, tau[i] );

        // Apply Householder reflection from the right to A[0:ihi,i+1:ihi]
        auto w = slice( work, pair{0,ihi} );
        auto C = slice( A, pair{0,ihi}, pair{i+1,ihi} );
        larf( right_side, v, tau[i], C, w );

        // Apply Householder reflection from the left to A[i+1:ihi,i+1:n-1]
        w = slice( work, pair{i+1,n} );
        C = slice( A, pair{i+1,ihi}, pair{i+1,n} );
        auto tauconj = conj(tau[i]);
        larf( left_side, v, tauconj, C, w );
	}

    return 0;
}

} // lapack

#endif // __GEHD2_HH__
