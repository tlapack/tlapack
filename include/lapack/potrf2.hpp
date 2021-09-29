/// @file potrf2.hpp Computes the Cholesky factorization of a Hermitian positive definite matrix A using the recursive algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __POTRF2_HH__
#define __POTRF2_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"
#include "lapack/potrf.hpp"
#include "tblas.hpp"

namespace lapack {

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
 * with n1 = n/2 and n2 = n-n1.
 * The subroutine calls itself to factor $A_{11},$
 * updates and scales $A_{21}$ or $A_{12},$
 * updates $A_{22},$
 * and calls itself to factor $A_{22}.$
 *
 * @param[in] uplo
 *     - lapack::Uplo::Upper: Upper triangle of A is stored;
 *     - lapack::Uplo::Lower: Lower triangle of A is stored.
 *
 * @param[in] n
 *     The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *     The n-by-n matrix A, stored in an lda-by-n array.
 *     On entry, the Hermitian matrix A.
 *     - If uplo = Upper, the leading
 *     n-by-n upper triangular part of A contains the upper
 *     triangular part of the matrix A, and the strictly lower
 *     triangular part of A is not referenced.
 *
 *     - If uplo = Lower, the
 *     leading n-by-n lower triangular part of A contains the lower
 *     triangular part of the matrix A, and the strictly upper
 *     triangular part of A is not referenced.
 *
 *     - On successful exit, the factor U or L from the Cholesky
 *     factorization $A = U^H U$ or $A = L L^H.$
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= max(1,n).
 *
 * @return = 0: successful exit
 * @return > 0: if return value = i, the leading minor of order i is not
 *     positive definite, and the factorization could not be
 *     completed.
 *
 * @ingroup posv_computational
 */
template< typename T >
int potrf2( Uplo uplo, Matrix< T > const A )
{
    typedef blas::real_type<T> real_t;
    using blas::trsm;
    using blas::syrk;
    using blas::sqrt;
    using blas::ColMajorMapping;
    using blas::matrix_extents;

    // Constants
    const real_t    rone( 1.0 );
    const T         one( 1.0 );
    const real_t    zero( 0.0 );

    // Check arguments
    lapack_error_if( uplo != Uplo::Upper && uplo != Uplo::Lower, -1 );
    lapack_error_if( n < 0, -2 );
    lapack_error_if( lda < n, -4 );

    // Quick return
    if (n == 0)
        return 0;

    // Stop recursion
    if (n == 1) {
        const real_t a00 = real( A(0,0) );
        if( a00 > zero ) {
            A(0,0) = sqrt( a00 );
            return 0;
        }
        else
            return 1;
    }
    // Recursive code
    {
        const idx_t n1 = n/2;
        const idx_t n2 = n-n1;

        // Define A11 and A22
        auto A11 = submatrix( A, std::pair(0,n1), std::pair(0,n1) );
        auto A22 = submatrix( A, std::pair(n1,n), std::pair(n1,n) );
        
        // Factor A11
        int info = potrf( uplo, A11 );
        if( info != 0 )
            return info;

        if( uplo == Uplo::Upper ) {

            // Update and scale A12
            auto A12 = submatrix( A, std::pair(0,n1), std::pair(n1,n) );
            trsm(
                Side::Left, Uplo::Upper,
                Op::ConjTrans, Diag::NonUnit,
                one, A11, A12 );

            // Update A22
            herk(
                uplo, Op::ConjTrans,
                -one, A12, one, A22 );
        }
        else {

            // Update and scale A12
            auto A12 = submatrix( A, std::pair(n1,n), std::pair(0,n1) );
            trsm(
                Side::Right, Uplo::Lower,
                Op::ConjTrans, Diag::NonUnit,
                one, A11, A12 );

            // Update A22
            herk(
                uplo, Op::NoTrans,
                -one, A12, one, A22 );
        }
        
        // Factor A22
        return potrf( uplo, A22 );
    }
}

} // lapack

#endif // __POTRF2_HH__