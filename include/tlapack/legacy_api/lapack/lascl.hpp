/// @file lascl.hpp Multiplies a matrix by a scalar.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LASCL_HH
#define TLAPACK_LEGACY_LASCL_HH

#include "tlapack/base/types.hpp"
#include "tlapack/lapack/lascl.hpp"

namespace tlapack {

/** @brief  Multiplies a matrix A by the real scalar a/b.
 *
 * Multiplication of a matrix A by scalar a/b is done without over/underflow as long as the final
 * result $a A/b$ does not over/underflow. The parameter type specifies that
 * A may be full, upper triangular, lower triangular, upper Hessenberg, or banded.
 * 
 * @return 0 if success.
 * @return -i if the ith argument is invalid.
 * 
 * @param[in] matrixtype Specifies the type of matrix A.
 *
 *        MatrixType::General: 
 *          A is a full matrix.
 *        MatrixType::Lower:
 *          A is a lower triangular matrix.
 *        MatrixType::Upper:
 *          A is an upper triangular matrix.
 *        MatrixType::Hessenberg:
 *          A is an upper Hessenberg matrix.
 *        MatrixType::LowerBand:
 *          A is a symmetric band matrix with lower bandwidth kl and upper bandwidth ku
 *          and with the only the lower half stored.
 *        MatrixType::UpperBand: 
 *          A is a symmetric band matrix with lower bandwidth kl and upper bandwidth ku
 *          and with the only the upper half stored.
 *        MatrixType::Band:
 *          A is a band matrix with lower bandwidth kl and upper bandwidth ku.
 * 
 * @param[in] kl The lower bandwidth of A, used only for banded matrix types B, Q and Z.
 * @param[in] ku The upper bandwidth of A, used only for banded matrix types B, Q and Z.
 * @param[in] b The denominator of the scalar a/b.
 * @param[in] a The numerator of the scalar a/b.
 * @param[in] m The number of rows of the matrix A. m>=0
 * @param[in] n The number of columns of the matrix A. n>=0
 * @param[in,out] A Pointer to the matrix A [in/out].
 * @param[in] lda The column length of the matrix A.
 * 
 * @ingroup legacy_lapack
 */
template< class matrixType_t, typename T >
int lascl(
    matrixType_t matrixtype,
    idx_t kl, idx_t ku,
    const real_type<T>& b, const real_type<T>& a,
    idx_t m, idx_t n,
    T* A, idx_t lda )
{
    using internal::colmajor_matrix;
    using internal::banded_matrix;
    using std::max;
    
    // check arguments
    tlapack_check_false(
        (matrixtype != MatrixType::General) && 
        (matrixtype != MatrixType::Lower) && 
        (matrixtype != MatrixType::Upper) && 
        (matrixtype != MatrixType::Hessenberg) && 
        (matrixtype != MatrixType::LowerBand) && 
        (matrixtype != MatrixType::UpperBand) && 
        (matrixtype != MatrixType::Band) );
    tlapack_check_false( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand) || 
            (matrixtype == MatrixType::Band)
        ) && (
            (kl < 0) ||
            (kl > max(m-1, idx_t(0)))
        ) );
    tlapack_check_false( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand) || 
            (matrixtype == MatrixType::Band)
        ) && (
            (ku < 0) ||
            (ku > max(n-1, idx_t(0)))
        ) );
    tlapack_check_false( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand)
        ) && ( kl != ku ) );
    tlapack_check_false( m < 0 );
    tlapack_check_false( (lda < m) && (
        (matrixtype == MatrixType::General) || 
        (matrixtype == MatrixType::Lower) ||
        (matrixtype == MatrixType::Upper) ||
        (matrixtype == MatrixType::Hessenberg) ) );
    tlapack_check_false( (matrixtype == MatrixType::LowerBand) && (lda < kl + 1) );
    tlapack_check_false( (matrixtype == MatrixType::UpperBand) && (lda < ku + 1) );
    tlapack_check_false( (matrixtype == MatrixType::Band) && (lda < 2 * kl + ku + 1) );

    if (matrixtype == MatrixType::LowerBand)
    {
        auto A_ = banded_matrix<T>( A, m, n, kl, 0 );
        return lascl( write_policy(A_), b, a, A_ );
    }
    else if (matrixtype == MatrixType::UpperBand)
    {
        auto A_ = banded_matrix<T>( A, m, n, 0, ku );
        return lascl( write_policy(A_), b, a, A_ );
    }
    else if (matrixtype == MatrixType::Band)
    {
        auto A_ = banded_matrix<T>( A, m, n, kl, ku );
        return lascl( write_policy(A_), b, a, A_ );
    }
    else {
        auto A_ = colmajor_matrix<T>( A, m, n, lda );
        
        if (matrixtype == MatrixType::General)
        {
            return lascl( MatrixAccessPolicy::Dense, b, a, A_ );
        }
        else if (matrixtype == MatrixType::Lower)
        {
            return lascl( MatrixAccessPolicy::LowerTriangle, b, a, A_ );
        }
        else if (matrixtype == MatrixType::Upper)
        {
            return lascl( MatrixAccessPolicy::UpperTriangle, b, a, A_ );
        }
        else // if (matrixtype == MatrixType::Hessenberg)
        {
            return lascl( MatrixAccessPolicy::UpperHessenberg, b, a, A_ );
        }
    }
}

}

#endif // TLAPACK_LEGACY_LASCL_HH
