/// @file getri_methodC.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
// PA = LU,   then A^(-1) is the solution to U (XP^T) L = I
#ifndef TLAPACK_GETRI_methodC_HH
#define TLAPACK_GETRI_methodC_HH

#include "tlapack/base/utils.hpp"
#include "tlapack.hpp"

namespace tlapack {
/** getri computes inverse of a general n-by-n matrix A
 *  using LU factorization. 
 *  we first run LU in place of A
 *  then we solve for X in the following equation
 * \[
 *   U X L = I
 * \]
 * Notice that from LU, we have PA=LU and as a result
 * \[
 *   U (A^{-1} P^{T}) L = I
 * \]
 * last equation means that $A^{-1} P^{T}=X$, therefore, to solve for $A^{-1}$
 * we just need to swap the columns of X according to $X=A^{-1} P$
 *
 * @return  0 if success
 * @return  -1 if matrix is not invertible
 *
 * @param[in,out] A n-by-n complex matrix.
 *      
 * @ingroup group_solve
 */
template< class matrix_t>
int getri_methodC( matrix_t& A){
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    
    // constant n, number of rows and also columns of A
    const idx_t n = ncols(A);

    // initialize a pivot vector and run getrf2 on A
    std::vector<idx_t> Piv( n , idx_t(0) );
    int not_inv=getrf2(A,Piv);
    
    // not_inv is the return integer of LU and if it is not zero, it means that our matrix is not invertible
    if(not_inv!=0){
        return -1;
    }

    // A has L and U in it, we will create X such that UXL=A in place of
    for(idx_t j=n-idx_t(1);j!=idx_t(-1);j--){
        if(j==n-1){
            // since A(n-1,n-1) is diagonal of U, if it zero, then the matrix is not invertible
            A(j,j) = T(1) / A(j,j);
            
        }
        else{
            // X22, l21, u12 are as in method C Nick Higham
            auto X22 = tlapack::slice(A,tlapack::range<idx_t>(j+1,n),tlapack::range<idx_t>(j+1,n));
            auto l21 = tlapack::slice(A,tlapack::range<idx_t>(j+1,n),j);
            auto u12 = tlapack::slice(A,j,tlapack::range<idx_t>(j+1,n));

            // first step of the algorithm, work1 holds x12
            // work1 = -X22 * l21
            std::vector<T> work1( n-j-idx_t(1) , T(0));
            tlapack::gemv(Op::NoTrans,T(-1), X22, l21,T(0), work1);
            
            //second line of the algorithm, work2 holds x21
            // work2 = -u12 X22 / A(j,j)
            std::vector<T> work2( n-j-idx_t(1) , T(0) );
            tlapack::gemv(Op::Trans,T(-1)/A(j,j), X22, u12,T(0), work2);

            // third line of the algorithm
            // A(j,j) = T(1) / A(j,j) - <x12,l21>
            A(j,j) = (T(1)/ A(j,j)) - tlapack::dotu(work2,l21);
            tlapack::copy(work1,l21);
            tlapack::copy(work2,u12);

        }
    }
    // swap columns of X to find A^{-1} since A^{-1}=X P
    for(idx_t j=n-idx_t(1);j!=idx_t(-1);j--){
                if(Piv[j]>j){
                    auto vect1=tlapack::col(A,j);
                    auto vect2=tlapack::col(A,Piv[j]);
                    tlapack::swap(vect1,vect2);
                }
        }
    
    return 0;
    
} // getri_methodC

} // lapack

#endif // TLAPACK_GETRI_methodC_HH



