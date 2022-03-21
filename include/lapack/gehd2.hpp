/// @file gehd2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __GEHD2_HH__
#define __GEHD2_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace lapack {

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
 *      0:ilo-1 and rows ihi:n-1 and is already upper triangular
 *      columns ihi:n-1 and rows 0:ilo-1.
 *      0 <= ilo <= ihi <= max(1,n-1).
 * @param[in,out] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 *      On exit, the upper triangle and the first subdiagonal of A
 *      are overwritten with the upper Hessenberg matrix H, and the
 *      elements below the first subdiagonal, with the array TAU,
 *      represent the orthogonal matrix Q as a product of elementary
 *      reflectors. See Further Details.
 * @param[out] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 * @param work Vector of size n-1.
 * 
 * @ingroup gehrd
 */
template< class matrix_t, class vector_t, class work_t >
int gehd2( size_type< matrix_t > ilo, size_type< matrix_t > ihi, matrix_t& A, vector_t &tau, work_t &work )
{
    using TA    = type_t< matrix_t >;
    using idx_t = size_type< matrix_t >;
    using pair  = std::pair<idx_t,idx_t>;

    // constants
    const TA one( 1 );
    const idx_t n = ncols(A);

    // check arguments
    lapack_error_if( ncols(A) != nrows(A), -1 );
    lapack_error_if( (idx_t) size(tau)  < n, -2 );
    lapack_error_if( (idx_t) size(work) < n-1, -3 );

    // quick return
    if (n <= 0) return 0;

    for(idx_t i = ilo; i < ihi-1; ++i) {

        // Define x := A[min(i+2,ihi):ihi,i]
        auto x = subvector( col( A, i ), pair{std::min<idx_t>(i+2,ihi),ihi} );

        // Generate the (i+1)-th elementary Householder reflection on x
        larfg( A(i+1,i), x, tau[i] );

        const auto alpha = A(i+1,i);
        A(i+1,i) = one;

        // Select v := A[i+1:ihi-1,i]
        const auto v = subvector( col( A, i ), pair{i+1,ihi-1} );
        // Apply Householder reflection from the right to A[0:ihi-1,i+1:ihi-1]
        auto C = submatrix( A, pair{0,ihi-1}, pair{i+1,ihi-1} );
        larf( right_side, v, tau[i], C, work );

        // Apply Householder reflection from the left to A[i+1:ihi-1,i+1:n-1]
        C = submatrix( A, pair{i+1,ihi-1}, pair{i+1,n-1} );
        larf( left_side, v, tau[i], C, work );

        A(i+1,i) = alpha;
	}

    return 0;
}

} // lapack

#endif // __GEHD2_HH__
