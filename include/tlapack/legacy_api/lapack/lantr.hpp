/// @file lantr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LANTR_HH
#define TLAPACK_LEGACY_LANTR_HH

#include "tlapack/base/types.hpp"
#include "tlapack/lapack/lantr.hpp"

namespace tlapack {

/** Calculates the value of the one norm, Frobenius norm, infinity norm, or element of largest absolute value of a triangular matrix
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
 * @param uplo Indicates whether A is upper or lower triangular.
 *      The other strict triangular part of A is not referenced.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 * 
 * @param m Number of rows to be included in the norm. m >= 0
 * @param n Number of columns to be included in the norm. n >= 0
 * @param A symmetric matrix size lda-by-n.
 * @param lda Leading dimension of matrix A.  ldA >= m
 * 
 * @ingroup legacy_lapack
**/
template <typename TA>
real_type<TA> lantr(
    Norm normType, Uplo uplo, Diag diag, idx_t m, idx_t n,
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
    tlapack_check_false( diag != Diag::NonUnit &&
                         diag != Diag::Unit );

    // quick return
    if (m == 0 || n == 0) return 0;

    // Matrix views
    auto A_ = colmajor_matrix<TA>( (TA*)A, m, n, lda );

    return lantr( normType, uplo, diag, A_ );
}

} // lapack

#endif // TLAPACK_LEGACY_LANTR_HH
