// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_HERK_HH
#define TLAPACK_BLAS_HERK_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Hermitian rank-k update:
 * \[
 *     C := \alpha A A^H + \beta C,
 * \]
 * or
 * \[
 *     C := \alpha A^H A + \beta C,
 * \]
 * where alpha and beta are real scalars, C is an n-by-n Hermitian matrix,
 * and A is an n-by-k or k-by-n matrix.
 *
 * Mind that if alpha or beta are complex,
 * the output matrix C may no longer Hermitian.
 *
 * @param[in] uplo
 *     What part of the matrix C is referenced,
 *     the opposite triangle being assumed from symmetry:
 *     - Uplo::Lower: only the lower triangular part of C is referenced.
 *     - Uplo::Upper: only the upper triangular part of C is referenced.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $C = \alpha A A^H + \beta C$.
 *     - Op::ConjTrans: $C = \alpha A^H A + \beta C$.
 *
 * @param[in] alpha Real scalar.
 * @param[in] A A n-by-k matrix.
 *     - If trans = NoTrans: a n-by-k matrix.
 *     - Otherwise:          a k-by-n matrix.
 * @param[in] beta Real scalar.
 * @param[in,out] C A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 *
 * @ingroup blas3
 */
template<
    class matrixA_t, class matrixC_t, 
    class alpha_t, class beta_t,
    enable_if_t<(
    /* Requires: */
        !is_complex<alpha_t>::value &&
        !is_complex<beta_t>::value
    ), int > = 0,
    class T  = type_t<matrixC_t>,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   real_type<T> >,
        pair< beta_t,    real_type<T> >
    > = 0
>
void herk(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A,
    const beta_t& beta, matrixC_t& C )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TC    = type_t< matrixC_t >;
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = (trans == Op::NoTrans) ? nrows(A) : ncols(A);
    const idx_t k = (trans == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    tlapack_check_false( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    tlapack_check_false( trans != Op::NoTrans &&
                   trans != Op::ConjTrans );
    tlapack_check_false( nrows(C) != ncols(C) );
    tlapack_check_false( nrows(C) != n );

    tlapack_check_false( access_denied( dense, read_policy(A) ) );
    tlapack_check_false( access_denied( uplo, write_policy(C) ) );

    if (trans == Op::NoTrans) {
        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {

                for(idx_t i = 0; i < j; ++i)
                    C(i,j) *= beta;
                C(j,j) = TC( beta * real(C(j,j)) );

                for(idx_t l = 0; l < k; ++l) {

                    auto alphaConjAjl = alpha*conj( A(j,l) );

                    for(idx_t i = 0; i < j; ++i)
                        C(i,j) += A(i,l)*alphaConjAjl;
                    C(j,j) += real(A(j,l)) * real(alphaConjAjl)
                            - imag(A(j,l)) * imag(alphaConjAjl);
                }
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {

                C(j,j) = TC( beta * real(C(j,j)) );
                for(idx_t i = j+1; i < n; ++i)
                    C(i,j) *= beta;

                for(idx_t l = 0; l < k; ++l) {

                    auto alphaConjAjl = alpha*conj( A(j,l) );

                    C(j,j) += real(A(j,l)) * real(alphaConjAjl)
                            - imag(A(j,l)) * imag(alphaConjAjl);
                    for(idx_t i = j+1; i < n; ++i)
                        C(i,j) += A(i,l) * alphaConjAjl;
                }
            }
        }
    }
    else { // trans == Op::ConjTrans
        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < j; ++i) {
                    TA sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += conj( A(l,i) ) * A(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
                real_type<TA> sum( 0 );
                for(idx_t l = 0; l < k; ++l)
                    sum += real(A(l,j)) * real(A(l,j))
                         + imag(A(l,j)) * imag(A(l,j));
                C(j,j) = alpha*sum + beta*real( C(j,j) );
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = j+1; i < n; ++i) {
                    TA sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += conj( A(l,i) ) * A(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
                real_type<TA> sum( 0 );
                for(idx_t l = 0; l < k; ++l)
                    sum += real(A(l,j)) * real(A(l,j))
                         + imag(A(l,j)) * imag(A(l,j));
                C(j,j) = alpha*sum + beta*real( C(j,j) );
            }
        }
    }

    if (uplo == Uplo::General) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = j+1; i < n; ++i)
                C(i,j) = conj( C(j,i) );
        }
    }
}

/**
 * Hermitian rank-k update:
 * \[
 *     C := \alpha A A^H,
 * \]
 * or
 * \[
 *     C := \alpha A^H A,
 * \]
 * where alpha and beta are real scalars, C is an n-by-n Hermitian matrix,
 * and A is an n-by-k or k-by-n matrix.
 *
 * Mind that if alpha or beta are complex,
 * the output matrix C may no longer Hermitian.
 *
 * @param[in] uplo
 *     What part of the matrix C is referenced,
 *     the opposite triangle being assumed from symmetry:
 *     - Uplo::Lower: only the lower triangular part of C is referenced.
 *     - Uplo::Upper: only the upper triangular part of C is referenced.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $C = \alpha A A^H$.
 *     - Op::ConjTrans: $C = \alpha A^H A$.
 *
 * @param[in] alpha Real scalar.
 * @param[in] A A n-by-k matrix.
 *     - If trans = NoTrans: a n-by-k matrix.
 *     - Otherwise:          a k-by-n matrix.
 * @param[out] C A n-by-n Hermitian matrix.
 *
 * @ingroup blas3
 */
template<
    class matrixA_t, class matrixC_t, 
    class alpha_t,
    enable_if_t<(
    /* Requires: */
        !is_complex<alpha_t>::value
    ), int > = 0,
    class T  = type_t<matrixC_t>,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   real_type<T> >
    > = 0
>
inline
void herk(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A,
    matrixC_t& C )
{
    return herk( uplo, trans, alpha, A, internal::StrongZero(), C );
}

#ifdef USE_LAPACKPP_WRAPPERS

    /**
     * Hermitian rank-k update
     * 
     * Wrapper to optimized BLAS.
     * 
     * @see herk(
        Uplo uplo,
        Op trans,
        const alpha_t& alpha, const matrixA_t& A,
        const beta_t& beta, matrixC_t& C )
    * 
    * @ingroup blas3
    */
    template<
        class matrixA_t, class matrixC_t, 
        class alpha_t, class beta_t,
        class T  = type_t<matrixC_t>,
        enable_if_allow_optblas_t<
            pair< matrixA_t, T >,
            pair< matrixC_t, T >,
            pair< alpha_t,   real_type<T> >,
            pair< beta_t,    real_type<T> >
        > = 0
    >
    inline
    void herk(
        Uplo uplo,
        Op trans,
        const alpha_t alpha, const matrixA_t& A,
        const beta_t beta, matrixC_t& C )
    {
        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto C_ = legacy_matrix(C);

        // Constants to forward
        const auto& n = C_.n;
        const auto& k = (trans == Op::NoTrans) ? A_.n : A_.m;

        if( alpha == alpha_t(0) )
            tlapack_warning( -3, "Infs and NaNs in A will not propagate to C on output" );
        if( beta == beta_t(0) )
            tlapack_warning( -5, "Infs and NaNs in C on input will not propagate to C on output" );

        return ::blas::herk(
            (::blas::Layout) A_.layout,
            (::blas::Uplo) uplo,
            (::blas::Op) trans, 
            n, k,
            alpha,
            A_.ptr, A_.ldim,
            beta,
            C_.ptr, C_.ldim );
    }

    /**
     * Hermitian rank-k update
     * 
     * Wrapper to optimized BLAS.
     * 
     * @see herk(
        Uplo uplo,
        Op trans,
        const alpha_t& alpha, const matrixA_t& A,
        matrixC_t& C )
    * 
    * @ingroup blas3
    */
    template<
        class matrixA_t, class matrixC_t, 
        class alpha_t,
        class T  = type_t<matrixC_t>,
        enable_if_allow_optblas_t<
            pair< matrixA_t, T >,
            pair< matrixC_t, T >,
            pair< alpha_t,   real_type<T> >
        > = 0
    >
    inline
    void herk(
        Uplo uplo,
        Op trans,
        const alpha_t alpha, const matrixA_t& A,
        matrixC_t& C )
    {
        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto C_ = legacy_matrix(C);

        // Constants to forward
        const auto& n = C_.n;
        const auto& k = (trans == Op::NoTrans) ? A_.n : A_.m;

        if( alpha == alpha_t(0) )
            tlapack_warning( -3, "Infs and NaNs in A or B will not propagate to C on output" );

        return ::blas::herk(
            (::blas::Layout) A_.layout,
            (::blas::Uplo) uplo,
            (::blas::Op) trans, 
            n, k,
            alpha,
            A_.ptr, A_.ldim,
            real_type<T>(0),
            C_.ptr, C_.ldim );
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_HERK_HH
