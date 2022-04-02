/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LARF_HH__
#define __LARF_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

#include "tblas.hpp"

namespace lapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 *
 * The elementary reflector H can be applied on either the left or right, with
 * \[
 *        H = I - \tau v v^H.
 * \]
 * If tau = 0, then H is taken to be the unit matrix.
 * 
 * v[0] is not accessed, and is instead assumed to be equal to one.
 * 
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 * 
 * @param[in] side Specifies whether the elementary reflector H is applied on the left or right.
 *
 *              side='L': form  H * C
 *              side='R': form  C * H
 * 
 * @param[in] m Number of rows of the matrix C.
 * @param[in] n Number of columns of the matrix C.
 * @param[in] v Vector of containing the elementary reflector.
 *
 *              If side='R', v is of length n.
 *              If side='L', v is of length m.
 * 
 * @param[in] incv Increment of the vector v.
 * @param[in] tau Value of tau in the representation of H.
 * @param[in,out] C m-by-n matrix.  On exit, C is overwritten with
 *
 *                H * C if side='L',
 *             or C * H if side='R'.
 * 
 * @param[in] ldC Column length of matrix C.  ldC >= m.
 * @param work Workspace vector of the following length:
 *
 *          n if side='L'
 *          m if side='R'.
 * 
 * @ingroup auxiliary
 */
template< class side_t, class vector_t, class tau_t, class matrix_t, class work_t >
inline void larf(
    side_t side,
    vector_t const& v, tau_t& tau,
    matrix_t& C, work_t& work )
{
    using blas::gemv;
    using blas::ger;
    using blas::copy;
    using blas::conj;

    // data traits
    using T = type_t<matrix_t>;
    using idx_t = size_type< matrix_t >;
    using pair = std::pair<size_t,size_t>;

    // constants
    const T one(1.0);
    const T zero(0.0);
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    // check arguments
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if(  access_denied( dense, write_policy(C) ) );

    // The following code was changed from:
    //
    // if( side == Side::Left ) {
    //     gemv(Op::NoTrans, one, C, v, zero, work);
    //     ger(-tau, work, v, C);
    // }
    // else{
    //     gemv(Op::ConjTrans, one, C, v, zero, work);
    //     ger(-tau, v, work, C);
    // }
    //
    // This is so that v[0] doesn't need to be changed to 1,
    // which is better for thread safety.

    if( side == Side::Left ) {
        auto w = subvector(work,pair{0,n});
        copy( row(C, 0), w );
        for (idx_t i = 0; i < n; ++i )
            w[i] = conj(w[i]);
        if(m > 1){
            auto x = subvector(v,pair{1,m});
            gemv(Op::ConjTrans, one, rows(C, pair{1,m}), x, one, w);
        }
        for (idx_t j = 0; j < n; ++j) {
            auto tmp = -tau * conj( w[j] );
            C(0,j) += tmp;
            for (idx_t i = 1; i < m; ++i)
                C(i,j) += v[i] * tmp;
        }
    }
    else {
        auto w = subvector(work,pair{0,m});
        copy( col(C, 0), w );
        if(n > 1){
            auto x = subvector(v,pair{1,n});
            gemv(Op::NoTrans, one, cols(C, pair{1,n}), x, one, w);
        }
        for (idx_t j = 0; j < n; ++j) {
            T tmp;
            if( j == 0 )
                tmp = -tau;
            else
                tmp = -tau * conj( v[j] );
            for (idx_t i = 0; i < m; ++i)
                C(i,j) += w[i] * tmp;
        }
    }
}

} // lapack

#endif // __LARF_HH__
