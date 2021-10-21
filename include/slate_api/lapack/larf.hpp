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

namespace lapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 * 
 * @see larf( Layout, Side side, blas::idx_t m, blas::idx_t n, TV const *v, blas::int_t incv, blas::scalar_type< TV, TC , TW > tau, TC *C, blas::idx_t ldC, TW *work )
 * 
 * @ingroup auxiliary
 */
template< typename TV, typename TC >
inline int larf(
    Side side,
    blas::idx_t m, blas::idx_t n,
    TV const *v, blas::int_t incv,
    blas::scalar_type< TV, TC > tau,
    TC *C, blas::idx_t ldC )
{
    typedef blas::scalar_type<TV, TC> scalar_t;
    scalar_t *work = new scalar_t[ ( side == Side::Left ) ? n : m ];
    int info;

    info = larf(
        Layout::ColMajor, side, m, n, v, incv, tau, C, ldC, work );
        
    delete[] work;

    return info;
}

} // lapack

#endif // __SLATE_LARF_HH__
