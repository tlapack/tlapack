/// @file potrs.hpp Apply the Cholesky factorization to solve a linear system.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __POTRS_HH__
#define __POTRS_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

#include "tblas.hpp"

namespace lapack {

/** Apply the Cholesky factorization to solve a linear system.
 * \[
 *      A X = B,
 * \]
 * where
 *      $A = U^H U,$ if uplo = Upper, or
 *      $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 * 
 * @tparam uplo_t
 *      Access type: Upper or Lower.
 *      Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A contains the matrix U;
 *      - Uplo::Lower: Lower triangle of A contains the matrix L.
 *      The other triangular part of A is not referenced.
 *
 * @param[in] A
 *      The factor U or L from the Cholesky factorization of A.
 *      
 *      - If uplo = Uplo::Upper, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = Uplo::Lower, the strictly upper
 *      triangular part of A is not referenced.
 *
 * @param[in,out] B
 *      On entry, the matrix B.
 *      On exit,  the matrix X.
 * 
 * @return = 0: successful exit.
 *
 * @ingroup posv_computational
 */
template< class uplo_t, class matrixA_t, class matrixB_t >
int potrs( uplo_t uplo, const matrixA_t& A, matrixB_t& B )
{
    using T = type_t< matrixB_t >;
    using blas::trsm;

    // Constants
    const T one( 1.0 );

    // Check arguments
    lapack_error_if(    uplo != Uplo::Lower &&
                        uplo != Uplo::Upper, -1 );
    lapack_error_if(    access_denied( uplo, write_policy(A) ), -1 );
    lapack_error_if(    nrows(A) != ncols(A), -2 );
    lapack_error_if(    nrows(B) != ncols(A), -3 );

    if( uplo == Uplo::Upper ) {
        // Solve A*X = B where A = U**H *U.
        trsm( left_side, uplo, conjTranspose, nonUnit_diagonal, one, A, B );
        trsm( left_side, uplo, noTranspose,   nonUnit_diagonal, one, A, B );
    }
    else {
        // Solve A*X = B where A = L*L**H.
        trsm( left_side, uplo, noTranspose,   nonUnit_diagonal, one, A, B );
        trsm( left_side, uplo, conjTranspose, nonUnit_diagonal, one, A, B );
    }
    return 0;
}

} // lapack

#endif // __POTRS_HH__