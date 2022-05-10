/// @file lacpy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lacpy.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_LACPY_HH__
#define __TLAPACK_LEGACY_LACPY_HH__

#include "lapack/lacpy.hpp"

namespace tlapack {

template< class uplo_t, typename TA, typename TB >
void lacpy(
    uplo_t uplo, idx_t m, idx_t n,
    const TA* A, idx_t lda,
    TB* B, idx_t ldb )
{
    using internal::colmajor_matrix;

    // check arguments
    tlapack_check_false(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper &&
                    uplo != Uplo::General );
    
    // Matrix views
    const auto A_ = colmajor_matrix<TA>( (TA*)A, m, n, lda );
    auto B_ = colmajor_matrix<TB>( B, m, n, ldb );

    lacpy( uplo, A_, B_ );
}

/** Copies a real matrix from A to B where A is either a full, upper triangular or lower triangular matrix.
 *
 * @param[in] matrixtype :
 *
 *        'U': A is assumed to be upper triangular; elements below the diagonal are not referenced.
 *        'L': A is assumed to be lower triangular; elements above the diagonal are not referenced.
 *        otherwise, A is assumed to be a full matrix.
 * 
 * @see lacpy( Uplo, idx_t, idx_t, TA*, idx_t, TB* B, idx_t )
 * 
 * @ingroup auxiliary
 */
template< typename TA, typename TB >
void inline lacpy(
    MatrixType matrixtype, idx_t m, idx_t n,
    const TA* A, idx_t lda,
    TB* B, idx_t ldb )
{
    if (matrixtype == MatrixType::Upper) {
        lacpy(Uplo::Upper, m, n, A, lda, B, ldb);
    } else if (matrixtype == MatrixType::Lower) {
        lacpy(Uplo::Lower, m, n, A, lda, B, ldb);
    } else {
        lacpy(Uplo::General, m, n, A, lda, B, ldb);
    }
}

}

#endif // __TLAPACK_LEGACY_LACPY_HH__
