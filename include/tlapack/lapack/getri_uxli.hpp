/// @file getri_uxli.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRI_UXLI_HH
#define TLAPACK_GETRI_UXLI_HH

#include "tlapack/base/utils.hpp"

#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/copy.hpp"

namespace tlapack {

template< class matrix_t >
inline constexpr
void getri_uxli_worksize( matrix_t& A, workinfo_t& workinfo, const workspace_opts_t<>& opts = {} )
{
    using T = type_t< matrix_t >;

    workinfo.m = sizeof(T);
    workinfo.n = ncols(A)-1;
}

/** getri computes inverse of a general n-by-n matrix A
 *  by solving for X in the following equation
 * \[
 *   U X L = I
 * \]
 *
 * @return = 0: successful exit
 * @return = i+1: if U(i,i) is exactly zero.  The triangular
 *          matrix is singular and its inverse can not be computed.
 *
 * @param[in,out] A n-by-n matrix.
 *      On entry, the factors L and U from the factorization P A = L U.
 *          L is stored in the lower triangle of A; unit diagonal is not stored.
 *          U is stored in the upper triangle of A.
 *      On exit, inverse of A is overwritten on A.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *      
 * @ingroup group_solve
 */
template< class matrix_t >
int getri_uxli( matrix_t& A, const workspace_opts_t<>& opts = {} )
{
    using work_t = vector_type< matrix_t, matrix_t >;
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;

    // Functor
    Create<work_t> new_vector;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    
    // constant n, number of rows and also columns of A
    const idx_t n = ncols(A);

    // Allocates workspace
    vectorOfBytes localworkdata;
    const Workspace work = [&]()
    {
        workinfo_t workinfo;
        getri_uxli_worksize( A, workinfo, opts );
        return alloc_workspace( localworkdata, workinfo, opts.work );
    }();
    auto w = new_vector( work, n-1 );

    // A has L and U in it, we will create X such that UXL=A in place of
    for(idx_t j=n-idx_t(1);j!=idx_t(-1);j--){
            
        // if A(j,j) is zero, then the matrix is not invertible
        if(A(j,j)==T(0))
            return j+1;
        
        if(j==n-1)
        {
            A(j,j) = T(1) / A(j,j);
        }
        else
        {    
            // X22, l21, u12 are as in method C Nick Higham
            auto X22 = tlapack::slice(A,tlapack::range<idx_t>(j+1,n),tlapack::range<idx_t>(j+1,n));
            auto l21 = tlapack::slice(A,tlapack::range<idx_t>(j+1,n),j);
            auto u12 = tlapack::slice(A,j,tlapack::range<idx_t>(j+1,n));
            auto slicework   = tlapack::slice(w,tlapack::range<idx_t>(0, n-j-1));

            // first step of the algorithm, work1 holds x12
            tlapack::gemv(Op::Trans,T(-1)/A(j,j), X22, u12,T(0), slicework);
            
            // second line of the algorithm, work2 holds x21
            A(j,j) = (T(1)/ A(j,j)) - tlapack::dotu(l21,slicework);

            // u12 updated, slicework available for use again
            tlapack::copy(slicework,u12);
            
            // third line of the algorithm
            tlapack::gemv(Op::NoTrans,T(-1), X22, l21,T(0), slicework);
            
            // update l21
            tlapack::copy(slicework,l21);

        }

    }
    
    return 0;
    
} // getri_uxli

} // lapack

#endif // TLAPACK_GETRI_UXLI_HH



