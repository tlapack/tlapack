/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_LARF_HH__
#define __SLATE_LARF_HH__

#include "lapack/larf.hpp"
#include <memory>

namespace lapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 * 
 * @see larf( Side side, blas::idx_t m, blas::idx_t n, TV const *v, blas::int_t incv, blas::scalar_type< TV, TC , TW > tau, TC *C, blas::idx_t ldC, TW *work )
 * 
 * @ingroup auxiliary
 */
template< typename TV, typename TC >
inline void larf(
    Side side,
    blas::idx_t m, blas::idx_t n,
    TV const *v, blas::int_t incv,
    blas::scalar_type< TV, TC > tau,
    TC *C, blas::idx_t ldC )
{
    typedef scalar_type<TV, TC> scalar_t;
    using blas::internal::colmajor_matrix;
    using blas::internal::vector;

    // check arguments
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incv == 0 );
    blas_error_if( ldC < m );

    // scalar_t *work = new scalar_t[ ( side == Side::Left ) ? n : m ];
    std::unique_ptr<scalar_t[]> work(new scalar_t[
        ( side == Side::Left ) ? n : m
    ]);

    // Initialize indexes
    idx_t lenv  = (( side == Side::Left ) ? m : n);
    idx_t lwork = (( side == Side::Left ) ? n : m);
    
    // Matrix views
    auto _C = colmajor_matrix<TC>( C, m, n, ldC );
    const auto _v = vector<TV>(
        (TV*) &v[(incv > 0 ? 0 : (-lenv + 1)*incv)],
        lenv, incv );
    auto _work = vector<scalar_t>( &work[0], lwork, 1 );

    if( side == Side::Left )
        larf( left_side, _v, tau, _C, _work);
    else
        larf( right_side, _v, tau, _C, _work);
        
    // delete[] work;
}

} // lapack

#endif // __SLATE_LARF_HH__
