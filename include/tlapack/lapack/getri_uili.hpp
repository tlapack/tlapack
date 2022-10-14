/// @file getri_uili.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_getri_uili_HH
#define TLAPACK_getri_uili_HH

#include "tlapack/base/utils.hpp"
#include <tlapack/lapack/getrf_recursive.hpp>
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/trtri_recursive.hpp"
#include "tlapack/blas/swap.hpp"

namespace tlapack {
/** getri_uili calculates inverse of a general n-by-n matrix A
 *  starts by computing L U factorization by getrf2 in place,
 *  then it uses trtri to invert U and L in place
 *  thereafter, ul_mult is called to calculate U^(-1)L^(-1) in place
 *  then columns of U^(-1)L^(-1) are swapped according to the pivot vector given by getrf_recursive
 *
 * @return  0 
 *
 * @param[in,out] A n-by-n complex matrix.
 *      
 *
 * @ingroup group_solve
 */
template< class matrix_t>
int getri_uili( matrix_t& A){
    
    using idx_t = size_type< matrix_t >;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    // quick return
    
    // constant, n is the number of rows and columns of the square matrix A
    const idx_t n = ncols(A);
    
    // call getrf_recursive to factorize Pivoted A to L and U in place
    std::vector<idx_t> Piv( n , idx_t(0) );
    getrf_recursive(A,Piv);

    // Invert the upper part of A; U
    trtri_recursive(Uplo::Upper, Diag::NonUnit, A);

    // Invert the lower part of A; L which has 1 on the diagonal
    trtri_recursive(Uplo::Lower, Diag::Unit, A);

    //multiply U^{-1} and L^{-1} in place using 
    ul_mult(A);
    
    // A <----- U^{-1}L^{-1}P; swapping columns of A according to Piv
    for(idx_t i=idx_t(n-1);i!=idx_t(-1);i--){
        if(Piv[i]!=i){
            auto vect1=tlapack::col(A,i);
            auto vect2=tlapack::col(A,Piv[i]);
            tlapack::swap(vect1,vect2);
        }
    }
    return 0;
    
} //getri_uili

} // lapack

#endif // TLAPACK_getri_uili_HH



