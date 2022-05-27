/// @file lansy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_LANSY_HH__
#define __TLAPACK_LEGACY_LANSY_HH__

#include <memory>

#include "base/types.hpp"
#include "lapack/lansy.hpp"

namespace tlapack {

/** Calculates the value of the one norm, Frobenius norm, infinity norm, or element of largest absolute value of a symmetric matrix
 *
 * @return Calculated norm value for the specified type.
 * 
 * @param normType Type should be specified as follows:
 *
 *     Norm::Max = maximum absolute value over all elements in A.
 *         Note: this is not a consistent matrix norm.
 *     Norm::One = one norm of the matrix A, the maximum value of the sums of each column.
 *     Norm::Inf = the infinity norm of the matrix A, the maximum value of the sum of each row.
 *     Norm::Fro = the Frobenius norm of the matrix A.
 *         This the square root of the sum of the squares of each element in A.
 * 
 * @param uplo Indicates whether the symmetric matrix A is stored as upper triangular or lower triangular.
 *      The other triangular part of A is not referenced.
 * @param n Number of columns to be included in the norm. n >= 0
 * @param A symmetric matrix size lda-by-n.
 * @param lda Leading dimension of matrix A.  ldA >= m
 * 
 * @ingroup auxiliary
**/
template <class norm_t, typename TA>
real_type<TA> lansy(
    norm_t normType, Uplo uplo, idx_t n,
    const TA *A, idx_t lda )
{
    using internal::colmajor_matrix;

    // check arguments
    tlapack_check_false(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    tlapack_check_false(  uplo != Uplo::Lower &&
                          uplo != Uplo::Upper );

    // quick return
    if ( n == 0 ) return 0;

    // Matrix views
    auto A_ = colmajor_matrix<TA>( (TA*)A, n, n, lda );

    return lansy( normType, uplo, A_ );
}

} // lapack

#endif // __TLAPACK_LEGACY_LANSY_HH__
