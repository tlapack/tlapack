/// @file orghr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dorghr.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __ORGHR_HH__
#define __ORGHR_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larf.hpp"
#include "lapack/org2r.hpp"

namespace lapack {

/** Generates a m-by-n matrix Q with orthogonal columns.
 * 
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      ilo and ihi must have the same values as in the
 *      previous call to gehrd. Q is equal to the unit
 *      matrix except in the submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *      0 <= ilo <= ihi <= max(1,n).
 * @param work Vector of size n-1.
 * 
 * @ingroup gehrd
 */
template<
    class matrix_t, class vector_t, class work_t >
int orghr(
    size_type< matrix_t > ilo,
    size_type< matrix_t > ihi,
    matrix_t& A,
    vector_t& tau,
    work_t& work )
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
    lapack_error_if( (idx_t) size(tau)  < std::min<idx_t>( m, n ), -2 );
    lapack_error_if( (idx_t) size(work) < n-1, -3 );


    // Shift the vectors which define the elementary reflectors one
    // column to the right, and set the first ilo and the last n-ihi
    // rows and columns to those of the unit matrix

    // This is currently optimised for column matrices, it may be interesting
    // to also write these loops for row matrices
    for(idx_t j = ihi-1; j > ilo; --j) {
        for(idx_t i = 0; i < j; ++i) {
            A(i,j) = zero;
        }
        for(idx_t i = j+1; i < ihi; ++i) {
            A(i,j) = A(i,j-1);
        }
        for(idx_t i = ihi; i < n; ++i ) {
            A(i,j) = zero;
        }
    }
    for(idx_t j = 0; j<ilo+1; ++j) {
        for(idx_t i = 0; i<n; ++i ) {
            A(i,j) = zero;
        }
        A(j,j) = one;
    }
    for(idx_t j = ihi; j<n; ++j ) {
        for(idx_t i = 0; i<n; ++i ) {
            A(i,j) = zero;
        }
        A(j,j) = one;
    }

    // Now that the vectors are shifted, we can call orgqr to generate the matrix
    // orgqr is not yet implemented, so we call org2r instead
    const idx_t nh = ihi-1-ilo;
    auto A_s = submatrix( A, pair{ilo+1,ihi}, pair{ilo+1,ihi} );
    auto tau_s = subvector( tau, pair{ilo,ihi-1} );
    org2r( nh, A_s, tau_s, work );

    return 0;
}

}

#endif // __ORGHR_HH__
