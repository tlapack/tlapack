/// @file potrs.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_POTRS_HH
#define TLAPACK_LEGACY_POTRS_HH

#include "tlapack/lapack/potrs.hpp"

namespace tlapack {

/** Apply the Cholesky factorization to solve a linear system.
 * 
 * @see potrs( uplo_t uplo, const matrixA_t& A, matrixB_t& B )
 * 
 * @ingroup posv_computational
 */
template< class uplo_t, typename T >
inline int potrs(
    uplo_t uplo, idx_t n, idx_t nrhs,
    const T* A, idx_t lda,
    T* B, idx_t ldb )
{
    using internal::colmajor_matrix;

    // Check arguments
    tlapack_check_false(    uplo != Uplo::Lower &&
                        uplo != Uplo::Upper );

    // Matrix views
    const auto A_ = colmajor_matrix<T>( (T*) A, n, n, lda );
          auto B_ = colmajor_matrix<T>( B, n, nrhs, ldb );

    return potrs( uplo, A_, B_ );
}

} // lapack

#endif // TLAPACK_LEGACY_POTRS_HH
