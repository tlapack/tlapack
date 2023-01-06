// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_HEMM_HH
#define TLAPACK_BLAS_HEMM_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Hermitian matrix-matrix multiply:
 * \[
 *     C := \alpha A B + \beta C,
 * \]
 * or
 * \[
 *     C := \alpha B A + \beta C,
 * \]
 * where alpha and beta are scalars, A is an m-by-m or n-by-n Hermitian matrix,
 * and B and C are m-by-n matrices.
 *
 * @param[in] side
 *     The side the matrix A appears on:
 *     - Side::Left:  $C = \alpha A B + \beta C$,
 *     - Side::Right: $C = \alpha B A + \beta C$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced:
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Scalar.
 * @param[in] A
 *     - If side = Left:  A m-by-m Hermitian matrix.
 *     - If side = Right: A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 * @param[in] B A m-by-n matrix.
 * @param[in] beta Scalar.
 * @param[in,out] C A m-by-n matrix.
 *
 * @ingroup blas3
 */
template<
    class matrixA_t,
    class matrixB_t,
    class matrixC_t, 
    class alpha_t,
    class beta_t,
    class T = type_t<matrixC_t>,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >,
        pair< beta_t,    T >
    > = 0
>
void hemm(
    Side side,
    Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t& beta, matrixC_t& C )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TB    = type_t< matrixB_t >;
    using idx_t = size_type< matrixB_t >;

    // using
    using scalar_t = scalar_type<TA,TB>;
            
    // constants
    const idx_t m = nrows(B);
    const idx_t n = ncols(B);

    // check arguments
    tlapack_check_false( side != Side::Left &&
                   side != Side::Right );
    tlapack_check_false( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    tlapack_check_false( nrows(A) != ncols(A) );
    tlapack_check_false( nrows(A) != ((side == Side::Left) ? m : n) );
    tlapack_check_false( nrows(C) != m );
    tlapack_check_false( ncols(C) != n );

    tlapack_check_false( access_denied( uplo,  read_policy(A) ) );
    tlapack_check_false( access_denied( dense, read_policy(B) ) );
    tlapack_check_false( access_denied( dense, write_policy(C) ) );

    if (side == Side::Left) {
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {

                    auto alphaTimesBij = alpha*B(i,j);
                    scalar_t sum( 0.0 );

                    for(idx_t k = 0; k < i; ++k) {
                        C(k,j) += A(k,i) * alphaTimesBij;
                        sum += conj( A(k,i) ) * B(k,j);
                    }
                    C(i,j) =
                        beta * C(i,j)
                        + real( A(i,i) ) * alphaTimesBij
                        + alpha * sum;
                }
            }
        }
        else {
            // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = m-1; i != idx_t(-1); --i) {

                    auto alphaTimesBij = alpha*B(i,j);
                    scalar_t sum( 0.0 );

                    for(idx_t k = i+1; k < m; ++k) {
                        C(k,j) += A(k,i) * alphaTimesBij;
                        sum += conj( A(k,i) ) * B(k,j);
                    }
                    C(i,j) =
                        beta * C(i,j)
                        + real( A(i,i) ) * alphaTimesBij
                        + alpha * sum;
                }
            }
        }
    }
    else { // side == Side::Right
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {

                auto alphaTimesAkj = alpha * real( A(j,j) );

                for(idx_t i = 0; i < m; ++i)
                    C(i,j) = beta * C(i,j) + B(i,j) * alphaTimesAkj;

                for(idx_t k = 0; k < j; ++k) {
                    alphaTimesAkj = alpha*A(k,j);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }

                for(idx_t k = j+1; k < n; ++k) {
                    alphaTimesAkj = alpha * conj( A(j,k) );
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }
            }
        }
        else {
            // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {

                auto alphaTimesAkj = alpha * real( A(j,j) );

                for(idx_t i = 0; i < m; ++i)
                    C(i,j) = beta * C(i,j) + B(i,j) * alphaTimesAkj;

                for(idx_t k = 0; k < j; ++k) {
                    alphaTimesAkj = alpha * conj( A(j,k) );
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }

                for(idx_t k = j+1; k < n; ++k) {
                    alphaTimesAkj = alpha*A(k,j);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += B(i,k) * alphaTimesAkj;
                }
            }
        }
    }
}

/**
 * Hermitian matrix-matrix multiply:
 * \[
 *     C := \alpha A B,
 * \]
 * or
 * \[
 *     C := \alpha B A,
 * \]
 * where alpha and beta are scalars, A is an m-by-m or n-by-n Hermitian matrix,
 * and B and C are m-by-n matrices.
 *
 * @param[in] side
 *     The side the matrix A appears on:
 *     - Side::Left:  $C = \alpha A B$,
 *     - Side::Right: $C = \alpha B A$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced:
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Scalar.
 * @param[in] A
 *     - If side = Left:  A m-by-m Hermitian matrix.
 *     - If side = Right: A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 * @param[in] B A m-by-n matrix.
 * @param[out] C A m-by-n matrix.
 *
 * @ingroup blas3
 */
template<
    class matrixA_t,
    class matrixB_t,
    class matrixC_t, 
    class alpha_t,
    class T = type_t<matrixC_t>,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >
    > = 0
>
inline
void hemm(
    Side side,
    Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    matrixC_t& C )
{
    return hemm( side, uplo, alpha, A, B, internal::StrongZero(), C );
}

#ifdef USE_LAPACKPP_WRAPPERS

    /**
     * Hermitian matrix-matrix multiply.
     * 
     * Wrapper to optimized BLAS.
     * 
     * @see hemm(
        Side side,
        Uplo uplo,
        const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
        const beta_t& beta, matrixC_t& C )
    * 
    * @ingroup blas3
    */
    template<
        class matrixA_t,
        class matrixB_t, 
        class matrixC_t, 
        class alpha_t, 
        class beta_t,
        class T  = type_t<matrixC_t>,
        enable_if_allow_optblas_t<
            pair< matrixA_t, T >,
            pair< matrixB_t, T >,
            pair< matrixC_t, T >,
            pair< alpha_t,   T >,
            pair< beta_t,    T >
        > = 0
    >
    inline
    void hemm(
        Side side,
        Uplo uplo,
        const alpha_t alpha, const matrixA_t& A, const matrixB_t& B,
        const beta_t beta, matrixC_t& C )
    {
        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto B_ = legacy_matrix(B);
        auto C_ = legacy_matrix(C);

        // Constants to forward
        const auto& m = C_.m;
        const auto& n = C_.n;

        if( alpha == alpha_t(0) )
            tlapack_warning( -3, "Infs and NaNs in A or B will not propagate to C on output" );
        if( beta == beta_t(0) )
            tlapack_warning( -6, "Infs and NaNs in C on input will not propagate to C on output" );

        return ::blas::hemm(
            (::blas::Layout) A_.layout,
            (::blas::Side) side, (::blas::Uplo) uplo, 
            m, n,
            alpha,
            A_.ptr, A_.ldim,
            B_.ptr, B_.ldim,
            beta,
            C_.ptr, C_.ldim );
    }

    /**
     * Hermitian matrix-matrix multiply.
     * 
     * Wrapper to optimized BLAS.
     * 
     * @see hemm(
        Side side,
        Uplo uplo,
        const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
        matrixC_t& C )
    * 
    * @ingroup blas3
    */
    template<
        class matrixA_t,
        class matrixB_t, 
        class matrixC_t, 
        class alpha_t,
        class T  = type_t<matrixC_t>,
        enable_if_allow_optblas_t<
            pair< matrixA_t, T >,
            pair< matrixB_t, T >,
            pair< matrixC_t, T >,
            pair< alpha_t,   T >
        > = 0
    >
    inline
    void hemm(
        Side side,
        Uplo uplo,
        const alpha_t alpha, const matrixA_t& A, const matrixB_t& B,
        matrixC_t& C )
    {
        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto B_ = legacy_matrix(B);
        auto C_ = legacy_matrix(C);

        // Constants to forward
        const auto& m = C_.m;
        const auto& n = C_.n;

        if( alpha == alpha_t(0) )
            tlapack_warning( -3, "Infs and NaNs in A or B will not propagate to C on output" );

        return ::blas::hemm(
            (::blas::Layout) A_.layout,
            (::blas::Side) side, (::blas::Uplo) uplo, 
            m, n,
            alpha,
            A_.ptr, A_.ldim,
            B_.ptr, B_.ldim,
            T(0),
            C_.ptr, C_.ldim );
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_HEMM_HH
