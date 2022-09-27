/// @file getri.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
// PA = LU   A^(-1)P^T=U^(-1)L^(-1) --- > U(A^(-1)P^T) L=I
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
    int res=getrf2(A,Piv);
    
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
            std::vector<T> work1( n-j-idx_t(1) , T(0));
            tlapack::gemv(Op::NoTrans,T(-1), X22, l21,T(0), work1);
            // std::cout<<nrows(X22)<<std::endl;
            
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
    for(idx_t j=n-idx_t(1);j!=idx_t(-1);j--){
                if(Piv[j]>j){
                    auto vect1=tlapack::col(A,j);
                    auto vect2=tlapack::col(A,Piv[j]);
                    tlapack::swap(vect1,vect2);
                }
        }
    // for(idx_t j=idx_t(0);j<n;j++){
    //             if(Piv[j]>j){
    //                 auto vect1=tlapack::row(A,j);
    //                 auto vect2=tlapack::row(A,Piv[j]);
    //                 tlapack::swap(vect1,vect2);
    //             }
    //     }
  


    
    
    return 0;
    
} // getrf2

} // lapack

#endif // TLAPACK_GETRI_HH



// if (m==1 || n ==1){
//         if(m==1){
//             Piv[0]=idx_t(0);
//                 if (A(Piv[0],0)==real_t(0)){
//                     return 1;
//                 }
//             return 0;
//         }
//         else{
//             idx_t toswap = idx_t(0);
//             toswap=iamax(tlapack::slice(A,tlapack::range<idx_t>(0,m),0));
//             Piv[0]=toswap;

//             if (Piv[0]!=0){
//                 auto vect1=tlapack::row(A,0);
//                 auto vect2=tlapack::row(A,toswap);
//                 tlapack::swap(vect1,vect2);
//             }
//             for(idx_t i=1;i<m;i++){
//                 A(i,0)/=A(0,0);
//             }
            
//             return 0;
//         }
        
//     }

// the case where m<n, we simply slice A into two parts, A0, a square matrix and A1 where A=[A0 , A1]
    // if(m<n){
    //     auto A0 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(0,m));
    //     auto A1 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(m,n));
    //     getri(A0,Piv);
    //     for(idx_t j=0;j<size(Piv);j++){
    //         if (Piv[j]>j){
    //             auto vect1=tlapack::row(A1,j);
    //             auto vect2=tlapack::row(A1,Piv[j]);
    //             tlapack::swap(vect1,vect2);
    //         }   
    //     }
    //     trsm(Side::Left,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A0,A1);
    //     return 0;
    // }
    // else{
    //     // Dimensions for the submatrices
    //     idx_t k0;
    //     k0=end/2;
    //     idx_t m1, n1;
    //     m1=m-k0;
    //     n1=n-k0;
        
    //     // in this step, we break A into two matrices, A=[A0 , A1]
    //     auto A0 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(0,k0));
    //     auto A1 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(k0,n));
        
    //     // Piv0 is the first k0 elements of Piv
    //     auto Piv0 = tlapack::slice(Piv,tlapack::range<idx_t>(0,k0));

    //     // Apply getrf2 on the left of half of the matrix
    //     getri(A0,Piv0);
        
    //     //swap the rows of A1
    //     for(idx_t j=0;j<size(Piv0);j++){
    //         if (Piv0[j]>j){
    //             auto vect1=tlapack::row(A1,j);
    //             auto vect2=tlapack::row(A1,Piv0[j]);
    //             tlapack::swap(vect1,vect2);
    //         }   
    //     }
        
    //     // Define the four blocks:
    //     //A00
    //     auto A00 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(0,k0));
    //     //A01
    //     auto A01 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(k0,n));
    //     //A10
    //     auto A10 = tlapack::slice(A,tlapack::range<idx_t>(k0,m),tlapack::range<idx_t>(0,k0));
    //     //A11
    //     auto A11 = tlapack::slice(A,tlapack::range<idx_t>(k0,m),tlapack::range<idx_t>(k0,n));

    //     // Take Piv1 to be the second slice of of Piv, meaning Piv= [Piv0, Piv1]
    //     auto Piv1 = tlapack::slice(Piv,tlapack::range<idx_t>(k0,end));

    //     // Solve the triangular system of equations given by A00 X = A01
    //     trsm(Side::Left,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A00,A01);
        
    //     // A11 <---- A11 - (A10 * A01)
    //     gemm(Op::NoTrans,Op::NoTrans,real_t(-1),A10,A01,real_t(1),A11);

    //     // Finding LU factorization of A11 in place
    //     getri(A11,Piv1);
        
    //     //swap the rows of A10 according to the swapped rows of A11 by refering to Piv1
    //     for(idx_t j=0;j<size(Piv1);j++){
    //         if (Piv1[j]>j){
    //             auto vect1=tlapack::row(A10,j);
    //             auto vect2=tlapack::row(A10,Piv1[j]);
    //             tlapack::swap(vect1,vect2);
    //         }   
    //     }
        
    //     // Shift Piv1, so Piv will have the accurate representation of overall pivots
    //     for(idx_t i=0;i<end-k0;i++){
    //         Piv1[i] += k0;
    //     }
        
    //     return 0;

    // }

    // for(idx_t j=idx_t(0);j<n;j++){
            //     if(Piv[j]>j){
            //         auto vect1=tlapack::col(A,j);
            //         auto vect2=tlapack::col(A,Piv[j]);
            //         tlapack::swap(vect1,vect2);
            //     }
            // }
            
            // for(idx_t j=idx_t(0);j<n;j++){
            //     auto vect1=tlapack::col(A,j);
            //     auto vect2=tlapack::col(A,Piv[j]);
            //     tlapack::swap(vect1,vect2);
            // }
            // for(idx_t j=idx_t(0);j<n;j++){
            //     auto vect1=tlapack::row(A,j);
            //     auto vect2=tlapack::row(A,Piv[j]);
            //     tlapack::swap(vect1,vect2);