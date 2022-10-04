/// @file ul_mult.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
// PA = LU   A^(-1)P^T=U^(-1)L^(-1) --- > U(A^(-1)P^T) L=I
#ifndef TLAPACK_ul_mult_HH
#define TLAPACK_ul_mult_HH

#include "tlapack/base/utils.hpp"
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
int ul_mult( matrix_t& A){
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    // quick return
    
    // constant
    const idx_t n = ncols(A);

    // if L and U are 1-by-1, then L is 1 and we simply UL=A(0,0)
    if(n==1){
        return -1;
        // return 0;
    }
    idx_t n0 = n / 2;
    // break A into four parts
    auto A00 = tlapack::slice(A,tlapack::range<idx_t>(0,n0),tlapack::range<idx_t>(0,n0));
    auto A10 = tlapack::slice(A,tlapack::range<idx_t>(n0,n),tlapack::range<idx_t>(0,n0));
    auto A01 = tlapack::slice(A,tlapack::range<idx_t>(0,n0),tlapack::range<idx_t>(n0,n));
    auto A11 = tlapack::slice(A,tlapack::range<idx_t>(n0,n),tlapack::range<idx_t>(n0,n));
    
    // calculate top left corner
    ul_mult(A00);
    tlapack::gemm(Op::NoTrans,Op::NoTrans,T(1),A01,A10,T(1),A00);

    // calculate bottom left corner
    tlapack::trmm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),A11,A10);
    
    // calculate top right
    tlapack::trmm(Side::Right,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A11,A01);
    
    // calculate bottom right
    ul_mult(A11);
    
    return 0;
    
} // ul_mult

} // lapack

#endif // TLAPACK_ul_mult_HH



