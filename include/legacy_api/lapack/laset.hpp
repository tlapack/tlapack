/// @file laset.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/laset.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_LASET_HH__
#define __TLAPACK_LEGACY_LASET_HH__

#include "lapack/laset.hpp"

namespace tlapack {

template< class uplo_t, typename TA >
void laset(
    uplo_t uplo, idx_t m, idx_t n,
    TA alpha, TA beta,
    TA* A, idx_t lda )
{
    using internal::colmajor_matrix;

    // check arguments
    tblas_error_if(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper &&
                    uplo != Uplo::General );

    // quick return
    if( m <= 0 || n <= 0 )
        return;
    
    // Matrix views
    auto A_ = colmajor_matrix<TA>( A, m, n, lda );

    return laset( uplo, alpha, beta, A_ );
}

/** Initializes a matrix to diagonal and off-diagonal values
 * 
 * @param[in] matrixtype :
 *
 *        'U': A is assumed to be upper triangular; elements below the diagonal are not referenced.
 *        'L': A is assumed to be lower triangular; elements above the diagonal are not referenced.
 *        otherwise, A is assumed to be a full matrix.
 *
 * @see laset( Uplo, idx_t, idx_t, TA, TA, TA*, idx_t )
 * 
 * @ingroup auxiliary
 */
template< typename TA >
void inline laset(
    MatrixType matrixtype, idx_t m, idx_t n,
    TA alpha, TA beta,
    TA* A, idx_t lda )
{
    if (matrixtype == MatrixType::Upper) {
        laset(Uplo::Upper, m, n, alpha, beta, A, lda);
    } else if (matrixtype == MatrixType::Lower) {
        laset(Uplo::Lower, m, n, alpha, beta, A, lda);
    } else {
        laset(Uplo::General, m, n, alpha, beta, A, lda);
    }
}

}

#endif // __TLAPACK_LEGACY_LASET_HH__
