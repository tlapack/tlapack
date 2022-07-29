/// @file unghr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dorghr.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGHR_HH
#define TLAPACK_UNGHR_HH

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larf.hpp"
#include "lapack/ung2r.hpp"

namespace tlapack {

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
int unghr(
    size_type< matrix_t > ilo,
    size_type< matrix_t > ihi,
    matrix_t& A,
    vector_t& tau,
    work_t& work )
{
    using T      = type_t< matrix_t >;
    using idx_t  = size_type< matrix_t >;
    using pair  = pair<idx_t,idx_t>;
    
    // constants
    const T zero( 0.0 );
    const T one ( 1.0 );
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t nh = ihi > ilo +1 ? ihi-1-ilo : 0;

    // check arguments
    tlapack_check_false( (idx_t) size(tau)  < std::min<idx_t>( m, n ) );
    tlapack_check_false( (idx_t) size(work) < n-1 );


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
    if( nh > 0 ){
        auto A_s = slice( A, pair{ilo+1,ihi}, pair{ilo+1,ihi} );
        auto tau_s = slice( tau, pair{ilo,ihi-1} );
        ung2r( nh, A_s, tau_s, work );
    }

    return 0;
}

}

#endif // TLAPACK_UNGHR_HH
