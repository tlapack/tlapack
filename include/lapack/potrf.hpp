/// @file potrf.hpp Computes the Cholesky factorization of a Hermitian positive definite matrix A using a blocked algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __POTRF_HH__
#define __POTRF_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

#include "lapack/potrf2.hpp"
#include "tblas.hpp"

namespace lapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using a blocked algorithm.
 *
 * The factorization has the form
 *      $A = U^H U,$ if uplo = Upper, or
 *      $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 * 
 * @tparam uplo_t   Either upper_triangle_t or lower_triangle_t.
 * @tparam matrix_t A \<T\>LAPACK abstract matrix.
 * @tparam opts_t
 * \code{.cpp}
 *      struct opts_t {
 *          size_type< matrix_t > nb; // Block size
 *          // ...
 *      };
 * \endcode
 *      If opts_t::nb does not exist, nb assumes a default value.
 *
 * @param[in] uplo
 *      - lapack::upper_triangle_t: Upper triangle of A is stored;
 *      - lapack::lower_triangle_t: Lower triangle of A is stored.
 *
 * @param[in,out] A
 *      On entry, the Hermitian matrix A.
 *      - If uplo = upper_triangle_t, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = lower_triangle_t, the strictly upper
 *      triangular part of A is not referenced.
 *
 *      - On successful exit, the factor U or L from the Cholesky
 *      factorization $A = U^H U$ or $A = L L^H.$
 *
 * @param[in,out] opts Options.
 *      - opts.nb Block size.
 *      If opts.nb does not exist or opts.nb <= 0, nb assumes a default value.
 *
 * @return = 0: successful exit
 * @return > 0: if return value = i, the leading minor of order i is not
 *      positive definite, and the factorization could not be completed.
 *
 * @ingroup posv_computational
 */
template< class uplo_t, class matrix_t, class opts_t >
int potrf( uplo_t uplo, matrix_t& A, opts_t&& opts )
{
    using T      = type_t< matrix_t >;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;
    using pair   = std::pair<idx_t,idx_t>;
    
    using blas::gemm;
    using blas::trsm;
    using blas::herk;
    using std::min;

    // Constants
    const real_t one( 1.0 );
    const idx_t n  = nrows(A);

    // Options
    const idx_t nb = get_nb(opts);

    // Check arguments
    lapack_error_if( nrows(A) != ncols(A), -2 );

    // Quick return
    if (n <= 0)
        return 0;

    // Unblocked code
    else if ( nb <= 1 && nb >= n)
        return potrf2( uplo, A );
    
    // Blocked code
    else {
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; j+=nb)
            {
                idx_t jb = min( nb, n-j );

                // Define AJJ and A1J
                auto AJJ = submatrix( A, pair{j,j+jb}, pair{j,j+jb} );
                auto A1J = submatrix( A, pair{0,j}, pair{j,j+jb} );

                herk( uplo, conjTranspose, -one, A1J, one, AJJ );
                
                int info = potrf2( uplo, AJJ );
                if( info != 0 )
                    return info + j;

                if( j+jb <= n ){

                    // Define B and C
                    auto B = submatrix( A, pair{0,j}, pair{j+jb,n} );
                    auto C = submatrix( A, pair{j,j+jb}, pair{j+jb,n} );
                
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
                auto AJJ = submatrix( A, pair{j,j+jb}, pair{j,j+jb} );
                auto AJ1 = submatrix( A, pair{j,j+jb}, pair{0,j} );

                herk( uplo, noTranspose, -one, AJ1, one, AJJ );
                
                int info = potrf2( uplo, AJJ );
                if( info != 0 )
                    return info + j;

                if( j+jb <= n ){

                    // Define B and C
                    auto B = submatrix( A, pair{j+jb,n}, pair{0,j} );
                    auto C = submatrix( A, pair{j+jb,n}, pair{j,j+jb} );
                
                    // Compute the current block row
                    gemm( noTranspose, conjTranspose, -one, B, AJ1, one, C );
                    trsm( right_side, uplo, conjTranspose, nonUnit_diagonal, one, AJJ, C );
                }
            }
        }
        return 0;
    }
}

} // lapack

#endif // __POTRF_HH__
