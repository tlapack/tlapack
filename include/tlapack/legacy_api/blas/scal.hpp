// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_SCAL_HH
#define TLAPACK_LEGACY_SCAL_HH

#include "tlapack/legacy_api/base/utils.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/blas/scal.hpp"

namespace tlapack {

/**
 * Scale vector by constant, $x = \alpha x$.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] n
 *     Number of elements in x. n >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*incx + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx > 0.
 *
 * @ingroup scal
 */
template< typename TA, typename TX >
void scal(
    idx_t n,
    const TA& alpha,
    TX* x, int_t incx )
{
    tlapack_check_false( incx <= 0 );

    // quick return
    if( n <= 0 ) return;
    
    tlapack_expr_with_vector_positiveInc(
        x_, TX, n, x, incx,
        return scal( alpha, x_ )
    );
}

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_LEGACY_SCAL_HH
