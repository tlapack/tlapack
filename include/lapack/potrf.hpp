/// @file potrf.hpp Computes the Cholesky factorization of a Hermitian positive definite matrix A using a blocked algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_POTRF_HH__
#define __TLAPACK_POTRF_HH__

#include "base/utils.hpp"

#include "lapack/potrf2.hpp"
#include "tblas.hpp"

namespace tlapack {

/// Default ptions for potrf
template< typename idx_t >
struct potrf_opts_t : public exception_t
{
    idx_t nb = 32; ///< Block size
};

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using a blocked algorithm.
 *
 * The factorization has the form
 *      $A = U^H U,$ if uplo = Upper, or
 *      $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 * 
 * @tparam uplo_t
 *      Access type: Upper or Lower.
 *      Either Uplo or any class that implements `operator Uplo()`.
 * 
 * @tparam opts_t
 *      Either potrf_opts_t or
 *      any struct that contains all potrf_opts_t members.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *      On entry, the Hermitian matrix A of size n-by-n.
 *      
 *      - If uplo = Uplo::Upper, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = Uplo::Lower, the strictly upper
 *      triangular part of A is not referenced.
 *
 *      - On successful exit, the factor U or L from the Cholesky
 *      factorization $A = U^H U$ or $A = L L^H.$
 *
 * @param[in] opts Options. @see potrf_opts_t.
 *
 * @return = 0: successful exit.
 * @return i, 0 < i <= n, if the leading minor of order i is not
 *      positive definite, and the factorization could not be completed.
 * @return n+1, if the factorization has some nans or infs.
 *
 * @ingroup posv_computational
 */
template< class uplo_t, class matrix_t, class opts_t >
int potrf( uplo_t uplo, matrix_t& A, opts_t&& opts )
{
    using T      = type_t< matrix_t >;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;
    using pair   = pair<idx_t,idx_t>;
    
    using std::min;

    // Constants
    const real_t one( 1.0 );
    const idx_t n  = nrows(A);
    const idx_t nb = opts.nb;

    // check arguments
    tlapack_assert( opts.paramCheck,  uplo == Uplo::Lower || uplo == Uplo::Upper, -1 );
    tlapack_assert( opts.accessCheck, access_granted( uplo, write_policy(A) ),    -2 );
    tlapack_assert( opts.sizeCheck,   nrows(A) == ncols(A),                       -2 );

    // Quick return
    if (n <= 0)
        return 0;

    // Unblocked code
    else if ( nb <= 1 || nb >= n )
        return potrf2( uplo, A );
    
    // Blocked code
    else {
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; j+=nb)
            {
                idx_t jb = min( nb, n-j );

                // Define AJJ and A1J
                auto AJJ = slice( A, pair{j,j+jb}, pair{j,j+jb} );
                auto A1J = slice( A, pair{0,j}, pair{j,j+jb} );

                herk( uplo, conjTranspose, -one, A1J, one, AJJ );
                
                int info = potrf2( uplo, AJJ );
                if( info != 0 ) {
                    tlapack_report( info + j,
                        "The leading minor of the reported order is not positive definite,"
                        " and the factorization could not be completed." );
                    return info + j;
                }

                if( j+jb < n ){

                    // Define B and C
                    auto B = slice( A, pair{0,j}, pair{j+jb,n} );
                    auto C = slice( A, pair{j,j+jb}, pair{j+jb,n} );
                
                    // Compute the current block row
                    gemm( conjTranspose, noTranspose, -one, A1J, B, one, C );
                    trsm( left_side, uplo, conjTranspose, nonUnit_diagonal, one, AJJ, C );
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; j+=nb)
            {
                idx_t jb = min( nb, n-j );

                // Define AJJ and AJ1
                auto AJJ = slice( A, pair{j,j+jb}, pair{j,j+jb} );
                auto AJ1 = slice( A, pair{j,j+jb}, pair{0,j} );

                herk( uplo, noTranspose, -one, AJ1, one, AJJ );
                
                int info = potrf2( uplo, AJJ );
                if( info != 0 ) {
                    tlapack_report( info + j,
                        "The leading minor of the reported order is not positive definite,"
                        " and the factorization could not be completed." );
                    return info + j;
                }

                if( j+jb < n ){

                    // Define B and C
                    auto B = slice( A, pair{j+jb,n}, pair{0,j} );
                    auto C = slice( A, pair{j+jb,n}, pair{j,j+jb} );
                
                    // Compute the current block row
                    gemm( noTranspose, conjTranspose, -one, B, AJ1, one, C );
                    trsm( right_side, uplo, conjTranspose, nonUnit_diagonal, one, AJJ, C );
                }
            }
        }

        // Report infs and nans on the output
        tlapack_report_nans_in_matrix( opts.nanCheck,
            uplo, A, n+1, "The factorization has some nans." );
        tlapack_report_infs_in_matrix( opts.infCheck,
            uplo, A, n+1, "The factorization has some infs." );
        
        return 0;
    }
}

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using a blocked algorithm.
 *
 * @see potrf( uplo_t uplo, matrix_t& A, opts_t&& opts ) 
 */
template< class uplo_t, class matrix_t >
int potrf( uplo_t uplo, matrix_t& A )
{    
    using idx_t = size_type< matrix_t >;
    return potrf( uplo, A, potrf_opts_t<idx_t>{} );
}

} // lapack

#endif // __POTRF_HH__