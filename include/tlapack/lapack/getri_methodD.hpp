/// @file getri_methodD.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
// PA = LU   A^(-1)P^T=U^(-1)L^(-1) --- > U(A^(-1)P^T) L=I
#ifndef TLAPACK_getri_methodD_HH
#define TLAPACK_getri_methodD_HH

#include "tlapack/base/utils.hpp"
#include <tlapack/lapack/getrf2.hpp>
#include "tlapack.hpp"

namespace tlapack {
/** test_ul computes UL of a general n-by-n matrix A
 *  where the nonzero part of L is the subdiagonal of A and on the diagonal of A is 1,
 *  nonzero part of U is diagonal and super-diagonal part of A 
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
    
    // constant
    const idx_t n = ncols(A);
    
    // LU factorize Pivoted A
    std::vector<idx_t> Piv( n , idx_t(0) );
    getrf2(A,Piv);
    
    // Invert the upper part of A, aka U
    trtri_recursive(Uplo::Upper, A);

    // save inverse of diagonal elements of U in work, and substitute with 1
    std::vector<T> work( n , T(0) );
    for (idx_t i = 0; i < n; ++i){
        work[i]=A(i,i);
        A(i,i)=T(1);
    }
    // Invert the lower part of A, aka L
    trtri_recursive(Uplo::Lower, A);

    //put inverse of diagonal elements of U back
    for (idx_t i = 0; i < n; ++i){
        A(i,i)=work[i];
    }

    //multiply U^{-1} and L^{-1} in place using 
    ul_mult(A);
    // A <----- U^{-1}L^{-1}P
    for(idx_t i=idx_t(n-1);i!=idx_t(-1);i--){
        auto vect1=tlapack::col(A,i);
        auto vect2=tlapack::col(A,Piv[i]);
        tlapack::swap(vect1,vect2);
    }
    return 0;
    
} //getri_methodD

} // lapack

#endif // TLAPACK_getri_methodD_HH



