/// @file potrf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_POTRF_HH
#define TLAPACK_LEGACY_POTRF_HH

#include "tlapack/lapack/potrf.hpp"

namespace tlapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using a blocked algorithm.
 * 
 * @see potrf( uplo_t uplo, matrix_t& A, const potrf_opts_t< size_type<matrix_t> >& opts = {} )
 * 
 * @ingroup legacy_lapack
 */
template< class uplo_t, typename T >
inline int potrf( uplo_t uplo, idx_t n, T* A, idx_t lda )
{
    using internal::colmajor_matrix;

    // check arguments
    tlapack_check_false(    uplo != Uplo::Lower &&
                        uplo != Uplo::Upper );

    // Matrix views
    auto A_ = colmajor_matrix( A, n, n, lda );

    return potrf_blocked( uplo, A_ );
}

} // lapack

#endif // TLAPACK_LEGACY_POTRF_HH
