/// @file lanhe.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lanhe.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LANHE_HH
#define TLAPACK_LANHE_HH

#include "tlapack/base/legacyArray.hpp"
#include "tlapack/lapack/lassq.hpp"

namespace tlapack {

/** Worspace query of lanhe().
 * 
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 * 
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 * 
 * @param[in] A n-by-n hermitian matrix.
 * 
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template< class norm_t, class uplo_t, class matrix_t >
inline constexpr
void lanhe_worksize(
    norm_t normType,
    uplo_t uplo,
    const matrix_t& A,
    workinfo_t& workinfo ) { }

/** Worspace query of lanhe().
 * 
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 * 
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 * 
 * @param[in] A n-by-n hermitian matrix.
 *
 * @param[in] opts Options.
 * 
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template< class norm_t, class uplo_t, class matrix_t >
inline constexpr
void lanhe_worksize(
    norm_t normType,
    uplo_t uplo,
    const matrix_t& A,
    workinfo_t& workinfo,
    const workspace_opts_t<>& opts )
{
    using T     = type_t< matrix_t >;

    if ( normType == Norm::Inf || normType == Norm::One )
    {
        const workinfo_t myWorkinfo( sizeof(T), nrows(A) );
        workinfo.minMax( myWorkinfo );
    }
}

/** Calculates the norm of a hermitian matrix.
 * 
 * @tparam norm_t Either Norm or any class that implements `operator Norm()`.
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 * 
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 * 
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 * 
 * @param[in] A n-by-n hermitian matrix.
 * 
 * @ingroup auxiliary
 */
template< class norm_t, class uplo_t, class matrix_t >
auto lanhe( norm_t normType, uplo_t uplo, const matrix_t& A )
{
    using T      = type_t<matrix_t>;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;
    using pair   = pair<idx_t,idx_t>;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check_false(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    tlapack_check_false(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper );
    tlapack_check_false(  access_denied( uplo, read_policy(A) ) );

    // quick return
    if ( n <= 0 ) return real_t( 0 );

    // Norm value
    real_t norm( 0 );

    if( normType == Norm::Max )
    {
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < j; ++i)
                {
                    real_t temp = tlapack::abs( A(i,j) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
                {
                    real_t temp = tlapack::abs( real(A(j,j)) );

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
                    real_t temp = tlapack::abs( real(A(j,j)) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
                for (idx_t i = j+1; i < n; ++i)
                {
                    real_t temp = tlapack::abs( A(i,j) );

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
                real_t temp( 0 );

                for (idx_t i = 0; i < j; ++i)
                    temp += tlapack::abs( A(i,j) );
                
                temp += tlapack::abs( real(A(j,j)) );

                for (idx_t i = j+1; i < n; ++i)
                    temp += tlapack::abs( A(j,i) );
                
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
                real_t temp( 0 );

                for (idx_t i = 0; i < j; ++i)
                    temp += tlapack::abs( A(j,i) );
                
                temp += tlapack::abs( real(A(j,j)) );

                for (idx_t i = j+1; i < n; ++i)
                    temp += tlapack::abs( A(i,j) );
                
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
                lassq( slice(A, pair{0,j}, j ), scale, ssq );
        }
        else {
            for (idx_t j = 0; j < n-1; ++j)
                lassq( slice(A, pair{j+1,n}, j ), scale, ssq );
        }
        ssq *= real_t(2);

        // Sum the real part in the diagonal
        lassq( diag(A,0), scale, ssq,
            // Lambda function to get the absolute value of the real part :
            []( const T& x ) { return tlapack::abs( real(x) ); }
        );

        // Compute the scaled square root
        norm = scale * sqrt(ssq);
    }

    return norm;
}

/** Calculates the norm of a hermitian matrix.
 * 
 * Code optimized for the infinity and one norm on column-major layouts using a workspace
 * of size at least n, where n is the number of rows of A.
 * @see lanhe( norm_t normType, uplo_t uplo, const matrix_t& A ).
 * 
 * @tparam norm_t Either Norm or any class that implements `operator Norm()`.
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 * 
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 * 
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 * 
 * @param[in] A n-by-n hermitian matrix.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 * 
 * @ingroup auxiliary
 */
template< class norm_t, class uplo_t, class matrix_t >
auto lanhe( norm_t normType, uplo_t uplo, const matrix_t& A, const workspace_opts_t<>& opts )
{
    using T      = type_t<matrix_t>;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;

    // check arguments
    tlapack_check_false(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    tlapack_check_false(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper );
    tlapack_check_false(  access_denied( uplo, read_policy(A) ) );

    // quick redirect for max-norm and Frobenius norm
    if      ( normType == Norm::Max  ) return lanhe( max_norm,  uplo, A );
    else if ( normType == Norm::Fro  ) return lanhe( frob_norm, uplo, A );
    else {

        // the code below uses a workspace and is meant for column-major layout
        // so as to do one pass on the data in a contiguous way when computing
        // the infinite and one norm

        // constants
        const idx_t n = nrows(A);

        // quick return
        if ( n <= 0 ) return real_t( 0 );

        // Allocates workspace
        vectorOfBytes localworkdata;
        const Workspace work = [&]()
        {
            workinfo_t workinfo;
            lanhe_worksize( normType, uplo, A, workinfo, opts );
            return alloc_workspace( localworkdata, workinfo, opts.work );
        }();
        auto w = legacyVector<T,idx_t>( n, work );

        // Norm value
        real_t norm( 0 );

        for (idx_t i = 0; i < n; ++i)
            w[i] = T(0);

        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; ++j)
            {
                real_t sum( 0 );
                for (idx_t i = 0; i < j; ++i) {
                    const real_t absa = tlapack::abs( A(i,j) );
                    sum += absa;
                    w[i] += absa;
                }
                w[j] = sum + tlapack::abs( real(A(j,j)) );
            }
            for (idx_t i = 0; i < n; ++i)
            {
                real_t sum = w[i];
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
                real_t sum = w[j] + tlapack::abs( real(A(j,j)) );
                for (idx_t i = j+1; i < n; ++i) {
                    const real_t absa = tlapack::abs( A(i,j) );
                    sum += absa;
                    w[i] += absa;
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
}

} // lapack

#endif // TLAPACK_LANHE_HH
