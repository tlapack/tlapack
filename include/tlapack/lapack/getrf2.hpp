/// @file getrf.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRF2_HH
#define TLAPACK_GETRF2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/swap.hpp"

namespace tlapack {

    /** getrf2 computes an LU factorization of a general m-by-n matrix A
     *  using partial pivoting with row interchanges. Recursive algorithm.
     *
     *  The factorization has the form
     * \[
     *   A = P L U
     * \]
     *  where P is a permutation matrix constructed from our Piv vector, L is lower triangular with unit
     *  diagonal elements (lower trapezoidal if m > n), and U is upper
     *  triangular (upper trapezoidal if m < n).
     *  This is Level 2 version of the algorithm.
     *
     * @return  0 if success
     * @return  i+1 if failed to compute the LU on iteration i
     *
     * @param[in,out] A m-by-n matrix.
     *      On exit, the factors L and U from the factorization A=PLU;
     *      the unit diagonal elements of L are not stored.
     *      
     * @param[in,out] Piv is a k-by-1 integer vector where k=min(m,n)
     * and Piv[i]=j where i<=j<=k-1, which means in the i-th iteration of the algorithm,
     * the j-th row needs to be swapped with i
     * 
     * @note To construct L and U, one proceeds as in the following steps
     *      1. Set matrices L m-by-k, and U k-by-n be to matrices with all zeros, where k=min(m,n)
     *      2. Set elements on the diagonal of L to 1
     *      3. below the diagonal of A will be copied to L
     *      4. On and above the diagonal of A will be copied to U
     *
     * @ingroup group_solve
     */
    template< class matrix_t, class vector_t >
    int getrf2( matrix_t& A, vector_t &Piv){
        
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
        if (m<=0 || n <= 0) return 0;
        // base case of recursion; one column matrices or one row matrices
        
        if (m==1 || n ==1){
            // one row matrices
            if(m==1){
                // Piv has one element
                Piv[0]=idx_t(0);
                if (A(Piv[0],0)==real_t(0)){
                    // in case which A(0,0) is zero, then we return 1 since in the first iteration we stopped
                    return 1;
                }
                return 0;
            }
            else{
                // when n==1, Piv has one element, Piv[0] needs to be swapped by the first row
                idx_t toswap = idx_t(0);
                toswap=iamax(tlapack::slice(A,tlapack::range<idx_t>(0,m),0));
                Piv[0]=toswap;
                
                // in the following case all elements are zero, and we return 1
                if (A(Piv[0],0)==real_t(0)){
                    return 1;
                }
                
                // in this case, we can safely swap since A(Piv[0],0) is not zero
                if (Piv[0]!=0){
                    auto vect1=tlapack::row(A,0);
                    auto vect2=tlapack::row(A,toswap);
                    tlapack::swap(vect1,vect2);
                }
                
                // by the previous comment, we can safely divide by A(0,0) and finish the base case of the algorithm
                for(idx_t i=1;i<m;i++){
                    A(i,0)/=A(0,0);
                }
                
                return 0;
            }
            
        }
        
        // the case where m<n, we simply slice A into two parts, A0, a square matrix and A1 where A=[A0 , A1]
        if(m<n){
            auto A0 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(0,m));
            auto A1 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(m,n));
            getrf2(A0,Piv);
            
            // swap the rows of A1 according to Piv
            for(idx_t j=0;j<size(Piv);j++){
                if (Piv[j]>j){
                    auto vect1=tlapack::row(A1,j);
                    auto vect2=tlapack::row(A1,Piv[j]);
                    tlapack::swap(vect1,vect2);
                }   
            }
            
            // Solve triangular system A0 X = A1 and update A1
            trsm(Side::Left,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A0,A1);
            return 0;
        }
        else{
            // Dimensions for the submatrices
            idx_t k0;
            k0=end/2;
            idx_t m1, n1;
            m1=m-k0;
            n1=n-k0;
            
            // in this step, we break A into two matrices, A=[A0 , A1]
            auto A0 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(0,k0));
            auto A1 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(k0,n));
            
            // Piv0 is the first k0 elements of Piv
            auto Piv0 = tlapack::slice(Piv,tlapack::range<idx_t>(0,k0));

            // Apply getrf2 on the left of half of the matrix
            getrf2(A0,Piv0);
            
            //swap the rows of A1
            for(idx_t j=0;j<size(Piv0);j++){
                if (Piv0[j]>j){
                    auto vect1=tlapack::row(A1,j);
                    auto vect2=tlapack::row(A1,Piv0[j]);
                    tlapack::swap(vect1,vect2);
                }   
            }
            
            // partition A into the following four blocks:
            //A00
            auto A00 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(0,k0));
            
            //A01
            auto A01 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(k0,n));
            
            //A10
            auto A10 = tlapack::slice(A,tlapack::range<idx_t>(k0,m),tlapack::range<idx_t>(0,k0));
            
            //A11
            auto A11 = tlapack::slice(A,tlapack::range<idx_t>(k0,m),tlapack::range<idx_t>(k0,n));

            // Take Piv1 to be the second slice of of Piv, meaning Piv= [Piv0, Piv1]
            auto Piv1 = tlapack::slice(Piv,tlapack::range<idx_t>(k0,end));

            // Solve the triangular system of equations given by A00 X = A01
            trsm(Side::Left,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A00,A01);
            
            // A11 <---- A11 - (A10 * A01)
            gemm(Op::NoTrans,Op::NoTrans,real_t(-1),A10,A01,real_t(1),A11);

            // Finding LU factorization of A11 in place
            getrf2(A11,Piv1);
            
            //swap the rows of A10 according to the swapped rows of A11 by refering to Piv1
            for(idx_t j=0;j<size(Piv1);j++){
                if (Piv1[j]>j){
                    auto vect1=tlapack::row(A10,j);
                    auto vect2=tlapack::row(A10,Piv1[j]);
                    tlapack::swap(vect1,vect2);
                }   
            }
            
            // Shift Piv1, so Piv will have the accurate representation of overall pivots
            for(idx_t i=0;i<end-k0;i++){
                Piv1[i] += k0;
            }
            
            return 0;

        }
        
    } // getrf2

} // tlapack

#endif // TLAPACK_GETRF2_HH
