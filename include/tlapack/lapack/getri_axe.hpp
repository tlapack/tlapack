/// @file getri_axe.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_getri_axe_HH
#define TLAPACK_getri_axe_HH

#include <iostream>
#include <stdio.h>
#include "tlapack/base/utils.hpp"
#include <tlapack/lapack/getrf_recursive.hpp>
#include <tlapack/blas/trsv.hpp>
#include "tlapack/blas/swap.hpp"

namespace tlapack {
/** getri_axe computes the inverse of a general n-by-n matrix A
 *  it computes L U factorization by getrf_recursive in place,
 *  thereafter, we solve L U (X) = I one column at a time
 *  then we find A^{-1} through A^{-1}= X P 
 * @return  0 
 *
 * @param[in,out] A n-by-n complex matrix.
 *      
 *
 * @ingroup group_solve
 */
template< class matrix_t>
int getri_axe( matrix_t& A){
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    // quick return
    
    // constant
    const idx_t n = ncols(A);
    
    // LU factorize Pivoted A
    std::vector<idx_t> Piv( n , idx_t(0) );
    getrf_recursive(A,Piv);

    // create X to store invese of A later
    std::vector<T> X_( n*n , T(0) );
    auto  X = tlapack::internal::colmajor_matrix<T>( &X_[0], n, n, n);
    
    for(idx_t i=idx_t(0); i<n; i++){
        X(i,i)=T(1);
        // to solve L U X_i = e_i, we will make i'th column of X to be e_i
        auto Xi = tlapack::slice(X,tlapack::range<idx_t>(0,n),i);
        
        // step1: solve L Y = e_i where e_i is a vector zeros with i'th position being 1.
        trsv(Uplo::Lower, Op::NoTrans, Diag::Unit, A, Xi);
        
        // step2: solve U X_i = Y
        trsv(Uplo::Upper, Op::NoTrans, Diag::NonUnit, A, Xi);
        
    }

    // copy inverse of X(inverse of A) into A
    for(idx_t i=idx_t(0); i<n; i++){
        for(idx_t j=idx_t(0); j<n; j++){
            A(i,j)=X(i,j);

        }
        
    }
    
    // A <----- U^{-1}L^{-1}P; swapping columns of A according to Piv
    for(idx_t i=idx_t(n-1);i!=idx_t(-1);i--){
        if(Piv[i]!=i){
            auto vect1=tlapack::col(A,i);
            auto vect2=tlapack::col(A,Piv[i]);
            tlapack::swap(vect1,vect2);
        }
    }
    return 0;
    
} //getri_axe

} // lapack

#endif // TLAPACK_getri_axe_HH



