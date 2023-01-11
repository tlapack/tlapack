/// @file hemv.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_HEMV_HH
#define TLAPACK_BLAS_HEMV_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Hermitian matrix-vector multiply:
 * \[
 *     y := \alpha A x + \beta y,
 * \]
 * where alpha and beta are scalars, x and y are vectors,
 * and A is an n-by-n Hermitian matrix.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed from symmetry.
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Scalar.
 * @param[in] A A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 * @param[in] x A n-element vector.
 * @param[in] beta Scalar.
 * @param[in,out] y A n-element vector.
 *
 * @ingroup blas2
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t, 
    class alpha_t, class beta_t,
    class T = type_t<vectorY_t>,
    disable_if_allow_optblas_t<
        pair< alpha_t,   T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >,
        pair< beta_t,    T >
    > = 0
>
void hemv(
    Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const vectorX_t& x,
    const beta_t& beta, vectorY_t& y )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TX    = type_t< vectorX_t >;
    using idx_t = size_type< matrixA_t >;

    // using
    using scalar_t = scalar_type<TA,TX>;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check_false( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    tlapack_check_false( ncols(A) != n );
    tlapack_check_false( size(x)  != n );
    tlapack_check_false( size(y)  != n );

    tlapack_check_false( access_denied( uplo, read_policy(A) ) );

    // form y = beta*y
    for (idx_t i = 0; i < n; ++i)
        y[i] *= beta;

    if (uplo == Uplo::Upper) {
        // A is stored in upper triangle
        // form y += alpha * A * x
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha*x[j];
            auto tmp2 = scalar_t(0);
            for (idx_t i = 0; i < j; ++i) {
                y[i] += tmp1 * A(i,j);
                tmp2 += conj( A(i,j) ) * x[i];
            }
            y[j] += tmp1 * real( A(j,j) ) + alpha * tmp2;
        }
    }
    else {
        // A is stored in lower triangle
        // form y += alpha * A * x
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha*x[j];
            auto tmp2 = scalar_t(0);
            for (idx_t i = j+1; i < n; ++i) {
                y[i] += tmp1 * A(i,j);
                tmp2 += conj( A(i,j) ) * x[i];
            }
            y[j] += tmp1 * real( A(j,j) ) + alpha * tmp2;
        }
    }
}

/**
 * Hermitian matrix-vector multiply:
 * \[
 *     y := \alpha A x,
 * \]
 * where alpha and beta are scalars, x and y are vectors,
 * and A is an n-by-n Hermitian matrix.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed from symmetry.
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Scalar.
 * @param[in] A A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 * @param[in] x A n-element vector.
 * @param[in,out] y A n-element vector.
 *
 * @ingroup blas2
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t, 
    class alpha_t,
    class T = type_t<vectorY_t>,
    disable_if_allow_optblas_t<
        pair< alpha_t,   T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void hemv(
    Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const vectorX_t& x,
    vectorY_t& y )
{
    return hemv( uplo, alpha, A, x, internal::StrongZero(), y );
}

#ifdef USE_LAPACKPP_WRAPPERS

    template<
        class matrixA_t,
        class vectorX_t, class vectorY_t, 
        class alpha_t, class beta_t,
        class T = type_t<vectorY_t>,
        enable_if_allow_optblas_t<
            pair< alpha_t, T >,
            pair< matrixA_t, T >,
            pair< vectorX_t, T >,
            pair< vectorY_t, T >,
            pair< beta_t,    T >
        > = 0
    >
    inline
    void hemv(
        Uplo uplo,
        const alpha_t alpha, const matrixA_t& A, const vectorX_t& x,
        const beta_t beta, vectorY_t& y )
    {
        using idx_t = size_type< matrixA_t >;

        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto x_ = legacy_vector(x);
        auto y_ = legacy_vector(y);

        // Constants to forward
        const idx_t& n = A_.n;
        const idx_t incx = (x_.direction == Direction::Forward) ? idx_t(x_.inc) : idx_t(-x_.inc);
        const idx_t incy = (y_.direction == Direction::Forward) ? idx_t(y_.inc) : idx_t(-y_.inc);

        if( alpha == alpha_t(0) )
            tlapack_warning( -2, "Infs and NaNs in A or x will not propagate to y on output" );
        if( beta == beta_t(0) )
            tlapack_warning( -5, "Infs and NaNs in y on input will not propagate to y on output" );

        return ::blas::hemv(
            (::blas::Layout) A_.layout,
            (::blas::Uplo) uplo,
            n,
            alpha,
            A_.ptr, A_.ldim,
            x_.ptr, incx,
            beta,
            y_.ptr, incy );
    }

    template<
        class matrixA_t,
        class vectorX_t, class vectorY_t, 
        class alpha_t,
        class T = type_t<vectorY_t>,
        enable_if_allow_optblas_t<
            pair< alpha_t, T >,
            pair< matrixA_t, T >,
            pair< vectorX_t, T >,
            pair< vectorY_t, T >
        > = 0
    >
    inline
    void hemv(
        Uplo uplo,
        const alpha_t alpha, const matrixA_t& A, const vectorX_t& x,
        vectorY_t& y )
    {
        using idx_t = size_type< matrixA_t >;

        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto x_ = legacy_vector(x);
        auto y_ = legacy_vector(y);

        // Constants to forward
        const idx_t& n = A_.n;
        const idx_t incx = (x_.direction == Direction::Forward) ? idx_t(x_.inc) : idx_t(-x_.inc);
        const idx_t incy = (y_.direction == Direction::Forward) ? idx_t(y_.inc) : idx_t(-y_.inc);

        if( alpha == alpha_t(0) )
            tlapack_warning( -2, "Infs and NaNs in A or x will not propagate to y on output" );

        return ::blas::hemv(
            (::blas::Layout) A_.layout,
            (::blas::Uplo) uplo,
            n,
            alpha,
            A_.ptr, A_.ldim,
            x_.ptr, incx,
            T(0),
            y_.ptr, incy );
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_HEMV_HH
