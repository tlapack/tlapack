/// @file getri_methodD.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
// PA = LU   A^(-1) = U^(-1)L^(-1)P
#ifndef TLAPACK_getri_methodD_HH
#define TLAPACK_getri_methodD_HH

#include "tlapack/base/utils.hpp"
#include <tlapack/lapack/getrf2.hpp>
#include "tlapack.hpp"

namespace tlapack {
/** getri_methodD calculates inverse of a general n-by-n matrix A
 *  starts by computing L U factorization by getrf2 in place,
 *  then it uses trtri to invert U and L in place
 *  thereafter, ul_mult is called to calculate U^(-1)L^(-1) in place
 *  then columns of U^(-1)L^(-1) are swapped according to the pivot vector given by getrf2
 *
 * @return  0 
 *
 * @param[in,out] A n-by-n complex matrix.
 *      
 *
 * @ingroup group_solve
 */
template< class matrix_t>
int getri_methodD( matrix_t& A){
    
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    // quick return
    
    // constant, n is the number of rows and columns of the square matrix A
    const idx_t n = ncols(A);
    
    // call getr2 to factorize Pivoted A to L and U in place
    std::vector<idx_t> Piv( n , idx_t(0) );
    getrf2(A,Piv);

    // Invert the upper part of A; U
    trtri_recursive(Uplo::Upper, Diag::NonUnit, A);

    // Invert the lower part of A; L which has 1 on the diagonal
    trtri_recursive(Uplo::Lower, Diag::Unit, A);

    //multiply U^{-1} and L^{-1} in place using 
    ul_mult(A);
    
    // A <----- U^{-1}L^{-1}P; swapping columns of A according to Piv
    for(idx_t i=idx_t(n-1);i!=idx_t(-1);i--){
        auto vect1=tlapack::col(A,i);
        auto vect2=tlapack::col(A,Piv[i]);
        tlapack::swap(vect1,vect2);
    }
    return 0;
    
} //getri_methodD

} // lapack

#endif // TLAPACK_getri_methodD_HH



