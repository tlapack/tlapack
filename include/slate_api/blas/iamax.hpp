// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_BLAS_IAMAX_HH
#define SLATE_BLAS_IAMAX_HH

#include "blas/utils.hpp"
#include "blas/iamax.hpp"

namespace blas {

/**
 * @brief Return $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$
 *
 * @param[in] n
 *     Number of elements in x.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*incx + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx > 0.
 * 
 * @return In priority order:
 * 1. 0 if n <= 0,
 * 2. the index of the first `NAN` in $x$ if it exists and if `checkNAN == true`,
 * 3. the index of the first `Infinity` in $x$ if it exists,
 * 4. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
 *     $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$.
 * 
 * @see iamax( check_t c, const vector_t& x )
 *
 * @ingroup iamax
 */
template< typename T >
inline idx_t
iamax(
    idx_t n, T const *x, blas::int_t incx )
{
    using internal::vector;

    // check arguments
    blas_error_if( incx <= 0 );

    const auto _x = vector<T>( (T*) x, n, incx );
    return iamax( _x );
}

}  // namespace blas

#endif        //  #ifndef SLATE_BLAS_IAMAX_HH
