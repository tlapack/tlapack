/// @file getrf.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRI_HH
#define TLAPACK_GETRI_HH

#include "tlapack/base/utils.hpp"
#include "tlapack.hpp"

namespace tlapack {
/** getrf computes an LU factorization of a general m-by-n matrix A
 *  using partial pivoting with row interchanges.
 *
 *  The factorization has the form
 * \[
 *   A = P L U
 * \]
 *  where P is a permutation matrix constructed from our Piv vector, L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 *  This is Level 0 version of the algorithm.
 *
 * @return  0 if success
 * @return  i+1 if failed to compute the LU on iteration i
 *
 * @param[in,out] A m-by-n complex matrix.
 *      A=PLU, and to construct L and U, one proceeds as in the following steps
 *      1. Set matrices L m-by-k, and U k-by-n be to matrices with all zeros, where k=min(m,n)
 *      2. Set elements on the diagonal of L to 1
 *      3. below the diagonal of A will be copied to L
 *      4. On and above the diagonal of A will be copied to U
 *      
 * @param[in,out] Piv is a k-by-1 vector where k=min(m,n)
 * and Piv[i]=j where i<=j<=k-1, which means in the i-th iteration of the algorithm,
 * the j-th row needs to be swapped with i
 *
 * @ingroup group_solve
 */
template< class matrix_t, class vector_t >
int getri( matrix_t& A, vector_t &Piv){
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t end = std::min<idx_t>( m, n );

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( (idx_t) size(Piv) >= end);
    // quick return
    if (m!=n) return 0;
    getrf2(A,Piv);
    

    // A has L and U in it, we will create X such that UXL=A in place of
    for(idx_t j=n-idx_t(1);j!=idx_t(-1);j--){
        if(j==n-1){
            A(j,j) = T(1) / A(j,j);
        }
        else{
            auto X22 = tlapack::slice(A,tlapack::range<idx_t>(j+1,n),tlapack::range<idx_t>(j+1,n));
            auto l21 = tlapack::slice(A,tlapack::range<idx_t>(j+1,n),j);
            auto u12 = tlapack::slice(A,j,tlapack::range<idx_t>(j+1,n));

            // first step of the algorithm, work1 holds x12
            // work1 = -X22 * l21
            std::vector<T> work1( n-j-id_t(1) , idx_t(0) );
            tlapack::gemv(Op::NoTrans,T(-1), X22, l21,T(1), work1);
            // std::cout<<nrows(X22)<<std::endl;
            
            //second line of the algorithm, work2 holds x21
            // work2 = -u12 X22 / A(j,j)
            std::vector<T> work2( n-j-id_t(1) , idx_t(0) );
            tlapack::gemv(Op::ConjTrans,T(-1)/A(j,j), X22, u12,T(1), work2);

            // third line of the algorithm
            // A(j,j) = T(1) / A(j,j) - <x12,l21>
            A(j,j) = T(1)/ A(j,j) - tlapack::dot(work2,l21);
            tlapack::copy(work1,l21);
            tlapack::copy(work2,u12);

            

        }
    }
    

    
    
    return 0;
    
} // getrf2

} // lapack

#endif // TLAPACK_GETRI_HH


