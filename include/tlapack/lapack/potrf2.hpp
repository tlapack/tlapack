/// @file potrf2.hpp Computes the Cholesky factorization of a Hermitian positive definite matrix A using the recursive algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_POTRF2_HH
#define TLAPACK_POTRF2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"

namespace tlapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using the recursive algorithm.
 *
 * The factorization has the form
 *     $A = U^H U,$ if uplo = Upper, or
 *     $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the recursive version of the algorithm. It divides
 * the matrix into four submatrices:
 * \[
 *     A = \begin{bmatrix}
 *             A_{11}  &  A_{12}
 *         \\  A_{21}  &  A_{22}
 *     \end{bmatrix}
 * \]
 * where $A_{11}$ is n1-by-n1 and $A_{22}$ is n2-by-n2,
 * with n1 = n/2 and n2 = n-n1, where n is the order of the matrix A.
 * The subroutine calls itself to factor $A_{11},$
 * updates and scales $A_{21}$ or $A_{12},$
 * updates $A_{22},$
 * and calls itself to factor $A_{22}.$
 * 
 * @tparam uplo_t
 *      Access type: Upper or Lower.
 *      Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *      On entry, the Hermitian matrix A.
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
 * @param[in] opts Options.
 *      Define the behavior of Exception Handling.   
 *
 * @return = 0: successful exit
 * @return > 0: if return value = i, the leading minor of order i is not
 *     positive definite, and the factorization could not be completed.
 *
 * @ingroup posv_computational
 */
template< class uplo_t, class matrix_t >
int potrf2( uplo_t uplo, matrix_t& A, const ec_opts_t& opts = {} )
{
    using T      = type_t< matrix_t >;
    using real_t = real_type<T>;
    using idx_t  = size_type< matrix_t >;
    using pair   = pair<idx_t,idx_t>;

    // Constants
    const real_t one( 1.0 );
    const real_t rzero( 0.0 );
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check_false(    uplo != Uplo::Lower &&
                            uplo != Uplo::Upper );
    tlapack_check_false(    access_denied( uplo, write_policy(A) ) );
    tlapack_check_false(    nrows(A) != ncols(A) );

    // Quick return
    if (n <= 0)
        return 0;

    // Stop recursion
    else if (n == 1) {
        const real_t a00 = real( A(0,0) );
        if( a00 > rzero ) {
            A(0,0) = sqrt( a00 );
            return 0;
        }
        else {
            tlapack_error_internal( opts.ec, 1,
                "The leading minor of order 1 is not positive definite,"
                " and the factorization could not be completed." );
            return 1;
        }
    }

    // Recursive code
    else {
        const idx_t n1 = n/2;

        // Define A11 and A22
        auto A11 = slice( A, pair{0,n1}, pair{0,n1} );
        auto A22 = slice( A, pair{n1,n}, pair{n1,n} );
        
        // Factor A11
        int info = potrf2( uplo, A11, noErrorCheck );
        if( info != 0 ) {
            tlapack_error_internal( opts.ec, info,
                "The leading minor of the reported order is not positive definite,"
                " and the factorization could not be completed." );
            return info;
        }

        if( uplo == Uplo::Upper ) {

            // Update and scale A12
            auto A12 = slice( A, pair{0,n1}, pair{n1,n} );
            trsm(
                Side::Left, Uplo::Upper,
                Op::ConjTrans, Diag::NonUnit,
                one, A11, A12 );

            // Update A22
            herk( uplo, Op::ConjTrans, -one, A12, one, A22 );
        }
        else {

            // Update and scale A21
            auto A21 = slice( A, pair{n1,n}, pair{0,n1} );
            trsm(
                Side::Right, Uplo::Lower,
                Op::ConjTrans, Diag::NonUnit,
                one, A11, A21 );

            // Update A22
            herk( uplo, Op::NoTrans, -one, A21, one, A22 );
        }
        
        // Factor A22
        info = potrf2( uplo, A22, noErrorCheck );
        if( info == 0 )
            return 0;
        else {
            tlapack_error_internal( opts.ec, info + n1,
                "The leading minor of the reported order is not positive definite,"
                " and the factorization could not be completed." );
            return info + n1;
        }
    }
}

} // lapack

#endif // TLAPACK_POTRF2_HH
