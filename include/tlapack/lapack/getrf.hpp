/// @file geqr2.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/geqr2.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRF_HH
#define TLAPACK_GETRF_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Computes a QR factorization of a matrix A.
 * 
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 * 
 * @return  0 if success
 * 
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *      If
 *          n-1 < m and
 *          type_t<matrix_t> == type_t<vector_t>
 *      then one may use tau[1:n] as the workspace.
 * 
 * @param work Vector of size n-1.
 * 
 * @ingroup geqrf
 */
template< class matrix_t >
int getrf( matrix_t& A, std::vector<idx_t> &P)
{
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    // using pair  = pair<idx_t,idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t end = std::min<idx_t>( m, n );

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    // tlapack_check_false( (idx_t) size(tau)  < std::min<idx_t>( m, n ) );
    // tlapack_check_false( (idx_t) size(work) < n-1 );

    // quick return
    idx_t toswap = idx_t(0);
    if (m<=0 || n <= 0) return 0;
    
    for(idx_t j=0;j<end;j++){
        toswap=j+iamax(tlapack::slice(A,tlapack::range<idx_t>(j,m),j));
        P[j]=toswap;
        auto vect1=tlapack::row(A,j);
        auto vect2=tlapack::row(A,toswap);
        tlapack::swap(vect1,vect2);
        // for (idx_t i = 0; i < m; ++i)
        // {
        //     std::cout << std::endl;
        //     for (idx_t k = 0; k < n; ++k)
        //         std::cout << std::setw(16) << A(i, k) << " ";
        // }
        // return pivot is zero
        if (A(j,j)==real_t(0)){
            return j+1;
        }


        
        
        
        for(idx_t i=j+1;i<m;i++){
            A(i,j)=A(i,j)/A(j,j);
        }
        for(idx_t row=j+1;row<m;row++){
            for(idx_t col=j+1;col<n;col++){
                A(row,col)=A(row,col)-A(row,j)*A(j,col);
            }   
        } 

    }
    
    return 0;
}

} // lapack

#endif // TLAPACK_GETRF_HH
//const idx_t toswap = iammax();
//tlapack::auto D=tlapack::slice(E,tlapack::range(m,m+1),tlapack::range(m,m+1));
