/// @file potrf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_POTRF_HH__
#define __TLAPACK_LEGACY_POTRF_HH__

#include "lapack/potrf.hpp"

namespace lapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using a blocked algorithm.
 * 
 * @see potrf( uplo_t uplo, matrix_t& A, opts_t&& opts )
 * 
 * @ingroup posv_computational
 */
template< typename T >
inline int potrf( Uplo uplo, idx_t n, T* A, idx_t lda )
{
    using blas::internal::colmajor_matrix;

    // Matrix views
    auto _A = colmajor_matrix<T>( A, n, n, lda );

    // Options
    struct { idx_t nb = 32; } opts;

    if( uplo == Uplo::Upper )
        return potrf( upper_triangle, _A, opts );
    else
        return potrf( lower_triangle, _A, opts );
}

} // lapack

#endif // __TLAPACK_LEGACY_POTRF_HH__
