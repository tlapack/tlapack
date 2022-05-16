/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_LARF_HH__
#define __TLAPACK_LEGACY_LARF_HH__

#include "lapack/larf.hpp"
#include <memory>

namespace tlapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 * 
 * @see larf( Side side, idx_t m, idx_t n, TV const *v, int_t incv, scalar_type< TV, TC , TW > tau, TC *C, idx_t ldC, TW *work )
 * 
 * @ingroup auxiliary
 */
template< class side_t, typename TV, typename TC >
inline void larf(
    side_t side,
    idx_t m, idx_t n,
    TV const *v, int_t incv,
    scalar_type< TV, TC > tau,
    TC *C, idx_t ldC )
{
    typedef scalar_type<TV, TC> scalar_t;
    using internal::colmajor_matrix;
    using internal::vector;

    // check arguments
    tblas_error_if( side != Side::Left &&
                   side != Side::Right );
    tblas_error_if( m < 0 );
    tblas_error_if( n < 0 );
    tblas_error_if( incv == 0 );
    tblas_error_if( ldC < m );

    // scalar_t *work = new scalar_t[ ( side == Side::Left ) ? n : m ];
    std::unique_ptr<scalar_t[]> work(new scalar_t[
        ( side == Side::Left ) ? n : m
    ]);

    // Initialize indexes
    idx_t lenv  = (( side == Side::Left ) ? m : n);
    idx_t lwork = (( side == Side::Left ) ? n : m);
    
    // Matrix views
    auto C_ = colmajor_matrix<TC>( C, m, n, ldC );
    auto _work = vector( &work[0], lwork );

    tlapack_expr_with_vector(
        _v, TV, lenv, v, incv,
        return larf( side, _v, tau, C_, _work)
    );
}

} // lapack

#endif // __TLAPACK_LEGACY_LARF_HH__
