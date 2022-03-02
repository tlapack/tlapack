/// @file nrm2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// Anderson E. (2017)
/// Algorithm 978: Safe Scaling in the Level 1 BLAS
/// ACM Trans Math Softw 44:1--28
/// @see https://doi.org/10.1145/3061665
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_NRM2_HH
#define TBLAS_LEGACY_NRM2_HH

#include "blas/utils.hpp"
#include "blas/nrm2.hpp"

namespace blas {

/**
 * @return 2-norm of vector,
 *     $|| x ||_2 = (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] n
 *     Number of elements in x. n >= 0.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*incx + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx > 0.
 *
 * @ingroup nrm2
 */
template< typename T >
real_type<T>
nrm2(
    blas::idx_t n,
    T const * x, blas::int_t incx )
{
    using internal::vector;

    // check arguments
    blas_error_if( incx <= 0 );

    const auto _x = vector<T>( (T*) x, n, incx );
    return nrm2( _x );
}

}  // namespace blas

#endif        // #ifndef TBLAS_LEGACY_NRM2_HH
