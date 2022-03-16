/// @file lanhe.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lanhe.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LANHE_HH__
#define __LANHE_HH__

#include "lapack/types.hpp"
#include "lapack/lassq.hpp"

namespace lapack {

/** Calculates the value of the one norm, Frobenius norm, infinity norm, or element of largest absolute value of a hermitian matrix
 *
 * @return Calculated norm value for the specified type.
 * 
 * @param normType Type should be specified as follows:
 *
 *     Norm::Max = maximum absolute value over all elements in A.
 *         Note: this is not a consistent matrix norm.
 *     Norm::One = one norm of the matrix A, the maximum value of the sums of each column.
 *     Norm::Inf = the infinity norm of the matrix A, the maximum value of the sum of each row.
 *     Norm::Fro = the Frobenius norm of the matrix A.
 *         This the square root of the sum of the squares of each element in A.
 * 
 * @param uplo Indicates whether the hermitian matrix A is stored as upper triangular or lower triangular.
 *      The other triangular part of A is not referenced.
 * @param n Number of columns to be included in the norm. n >= 0
 * @param A hermitian matrix size lda-by-n.
 * @param lda Leading dimension of matrix A.  ldA >= m
 * 
 * @ingroup auxiliary
**/
template< class norm_t, class uplo_t, class matrix_t >
real_type< type_t<matrix_t> >
lanhe( norm_t normType, uplo_t uplo, const matrix_t& A )
{
    using T      = type_t<matrix_t>;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;
    using pair   = std::pair<idx_t,idx_t>;
    using blas::isnan;
    using blas::sqrt;
    using blas::real;

    // constants
    const real_t zero(0.0);
    const idx_t n = nrows(A);

    // check arguments
    blas_error_if(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    blas_error_if(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper );

    // quick return
    if ( n <= 0 ) return zero;

    // Norm value
    real_t norm(0.0);

    if( normType == Norm::Max )
    {
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < j; ++i)
                {
                    real_t temp = blas::abs( A(i,j) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
                {
                    real_t temp = blas::abs( real(A(j,j)) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j) {
                {
                    real_t temp = blas::abs( real(A(j,j)) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
                for (idx_t i = j+1; i < n; ++i)
                {
                    real_t temp = blas::abs( A(i,j) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
            }
        }
    }
    else if( normType == Norm::One || normType == Norm::Inf )
    {
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; ++j) {
                real_t temp = 0;

                for (idx_t i = 0; i < j; ++i)
                    temp += blas::abs( A(i,j) );
                
                temp += blas::abs( real(A(j,j)) );

                for (idx_t i = j+1; i < n; ++i)
                    temp += blas::abs( A(j,i) );
                
                if (temp > norm)
                    norm = temp;
                else {
                    if ( isnan(temp) ) 
                        return temp;
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j) {
                real_t temp = 0;

                for (idx_t i = 0; i < j; ++i)
                    temp += blas::abs( A(j,i) );
                
                temp += blas::abs( real(A(j,j)) );

                for (idx_t i = j+1; i < n; ++i)
                    temp += blas::abs( A(i,j) );
                
                if (temp > norm)
                    norm = temp;
                else {
                    if ( isnan(temp) ) 
                        return temp;
                }
            }
        }
    }
    else
    {
        // Scaled ssq
        real_t scale(0), ssq(1);
        
        // Sum off-diagonals
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 1; j < n; ++j)
                lassq( subvector( col(A,j), pair{0,j} ), scale, ssq );
        }
        else {
            for (idx_t j = 0; j < n-1; ++j)
                lassq( subvector( col(A,j), pair{j+1,n} ), scale, ssq );
        }
        ssq *= 2;

        // Sum the real part in the diagonal
        struct absReValue {
            static inline real_t abs( const T& x ) {
                return blas::abs( real(x) );
            }
        };
        lassq< absReValue >( diag(A,0), scale, ssq );

        // Compute the scaled square root
        norm = scale * sqrt(ssq);
    }

    return norm;
}

template< class norm_t, class uplo_t, class matrix_t, class work_t >
real_type< type_t<matrix_t> >
lanhe( norm_t normType, uplo_t uplo, const matrix_t& A, work_t& work )
{
    using real_t = real_type< type_t<matrix_t> >;
    using idx_t  = size_type< matrix_t >;
    using blas::isnan;
    using blas::real;

    // check arguments
    blas_error_if(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    blas_error_if(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper );

    // quick redirect for max-norm and Frobenius norm
    if      ( normType == Norm::Max  ) return lanhe( max_norm,  uplo, A );
    else if ( normType == Norm::Fro  ) return lanhe( frob_norm, uplo, A );

    // the code below uses a workspace and is meant for column-major layout
    // so as to do one pass on the data in a contiguous way when computing
    // the infinite norm

    // constants
    const real_t zero(0.0);
    const idx_t n = nrows(A);

    // quick return
    if ( n <= 0 ) return zero;

    // Norm value
    real_t norm(0.0);

    for (idx_t i = 0; i < n; ++i)
        work[i] = type_t<work_t>(0);

    if( uplo == Uplo::Upper ) {
        for (idx_t j = 0; j < n; ++j)
        {
            real_t sum = zero;
            for (idx_t i = 0; i < j; ++i) {
                const real_t absa = blas::abs( A(i,j) );
                sum += absa;
                work[i] += absa;
            }
            work[j] = sum + blas::abs( real(A(j,j)) );
        }
        for (idx_t i = 0; i < n; ++i)
        {
            real_t sum = work[i];
            if (sum > norm)
                norm = sum;
            else {
                if ( isnan(sum) )
                    return sum;
            }
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j)
        {
            real_t sum = work[j] + blas::abs( real(A(j,j)) );
            for (idx_t i = j+1; i < n; ++i) {
                const real_t absa = blas::abs( A(i,j) );
                sum += absa;
                work[i] += absa;
            }
            if (sum > norm)
                norm = sum;
            else {
                if ( isnan(sum) )
                    return sum;
            }
        }
    }

    return norm;
}

} // lapack

#endif // __LANHE_HH__
