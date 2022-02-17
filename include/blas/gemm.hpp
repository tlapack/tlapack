// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_GEMM_HH
#define BLAS_GEMM_HH

#include "blas/utils.hpp"

#define TLAPACK_USE_BLAS

#define TLAPACK_OPT_TYPE( T ) \
    template<> struct has_blas_type< T > { \
        using type = T; \
        static constexpr bool value = true; \
    }

namespace blas {

    /// alias has_blas_type for array and constant types
    template<class...> struct has_blas_type : public std::false_type { };

    /// alias template for has_blas_type
    template<class... arrays_t>
    constexpr bool has_blas_type_v = has_blas_type< arrays_t... >::value;

    /// Optimized types
    TLAPACK_OPT_TYPE(float);
    TLAPACK_OPT_TYPE(double);
    TLAPACK_OPT_TYPE(std::complex<float>);
    TLAPACK_OPT_TYPE(std::complex<double>);

    /// alias has_blas_type for multiple arrays
    template<class array1_t, class array2_t, class... arrays_t>
    struct has_blas_type< array1_t, array2_t, arrays_t... > {
        using type = type_t<array1_t>;
        static constexpr bool value = 
            has_blas_type_v<array1_t> &&
            is_same_v< type, typename has_blas_type<array2_t,arrays_t...>::type >;
    };

    template<class array_t, class... arrays_t>
    using enable_if_has_blas_type_t = enable_if_t<(
    /* Requires: */
        has_blas_type_v< array_t, arrays_t... >
    ), int >;

    template<class array_t, class... arrays_t>
    using enable_if_hasnt_blas_type_t = enable_if_t<(
    /* Requires: */
        ! has_blas_type_v< array_t, arrays_t... >
    ), int >;
}

#ifndef TLAPACK_USE_BLAS
    // <T>LAPACK templates enabled for all data types
    #define _ENABLE_IF_HASNT_BLAS_TYPE(...) int
#else
    // <T>LAPACK templates enabled for all data types other than the ones in BLAS
    #define _ENABLE_IF_HAS_BLAS_TYPE(...) enable_if_has_blas_type_t< __VA_ARGS__ >
    #define _ENABLE_IF_HASNT_BLAS_TYPE(...) enable_if_hasnt_blas_type_t< __VA_ARGS__ >
#endif

namespace blas {

/**
 * General matrix-matrix multiply:
 * \[
 *     C = \alpha op(A) \times op(B) + \beta C,
 * \]
 * where $op(X)$ is one of
 *     $op(X) = X$,
 *     $op(X) = X^T$, or
 *     $op(X) = X^H$,
 * alpha and beta are scalars, and A, B, and C are matrices, with
 * $op(A)$ an m-by-k matrix, $op(B)$ a k-by-n matrix, and C an m-by-n matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] transA
 *     The operation $op(A)$ to be used:
 *     - Op::NoTrans:   $op(A) = A$.
 *     - Op::Trans:     $op(A) = A^T$.
 *     - Op::ConjTrans: $op(A) = A^H$.
 *
 * @param[in] transB
 *     The operation $op(B)$ to be used:
 *     - Op::NoTrans:   $op(B) = B$.
 *     - Op::Trans:     $op(B) = B^T$.
 *     - Op::ConjTrans: $op(B) = B^H$.
 *
 * @param[in] alpha scalar.
 * @param[in] A matrix.
 * @param[in] B matrix.
 * @param[in] beta scalar.
 * @param[in,out] C matrix.
 * 
 * @ingroup gemm
 */
template<
    class matrixA_t,
    class matrixB_t, 
    class matrixC_t, 
    class alpha_t, 
    class beta_t,
    _ENABLE_IF_HASNT_BLAS_TYPE(matrixA_t, matrixB_t, matrixC_t, alpha_t, beta_t) = 0
>
void gemm(
    Op transA,
    Op transB,
    const alpha_t& alpha,
    const matrixA_t& A,
    const matrixB_t& B,
    const beta_t& beta,
    matrixC_t& C )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TB    = type_t< matrixB_t >;
    using idx_t = size_type< matrixA_t >;

    // using
    using scalar_t = scalar_type<TA,TB>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = (transA == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if(
        m != ((transA == Op::NoTrans) ? nrows(A) : ncols(A)) );
    blas_error_if(
        n != ((transB == Op::NoTrans) ? ncols(B) : nrows(B)) );
    blas_error_if(
        ((transB == Op::NoTrans) ? nrows(B) : ncols(B)) != k );

    if (transA == Op::NoTrans) {
        if (transB == Op::NoTrans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    C(i,j) *= beta;
                for(idx_t l = 0; l < k; ++l) {
                    const auto alphaTimesblj = alpha*B(l,j);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += A(i,l)*alphaTimesblj;
                }
            }
        }
        else if (transB == Op::Trans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    C(i,j) *= beta;
                for(idx_t l = 0; l < k; ++l) {
                    const auto alphaTimesbjl = alpha*B(j,l);
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += A(i,l)*alphaTimesbjl;
                }
            }
        }
        else { // transB == Op::ConjTrans
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    C(i,j) *= beta;
                for(idx_t l = 0; l < k; ++l) {
                    const auto alphaTimesbjl = alpha*conj(B(j,l));
                    for(idx_t i = 0; i < m; ++i)
                        C(i,j) += A(i,l)*alphaTimesbjl;
                }
            }
        }
    }
    else if (transA == Op::Trans) {
        if (transB == Op::NoTrans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*B(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*B(j,l);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else { // transB == Op::ConjTrans
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*conj(B(j,l));
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
    }
    else { // transA == Op::ConjTrans
        if (transB == Op::NoTrans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += conj(A(l,i))*B(l,j);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += conj(A(l,i))*B(j,l);
                    C(i,j) = alpha*sum + beta*C(i,j);
                }
            }
        }
        else { // transB == Op::ConjTrans
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i) {
                    scalar_t sum( 0 );
                    for(idx_t l = 0; l < k; ++l)
                        sum += A(l,i)*B(j,l); // little improvement here
                    C(i,j) = alpha*conj(sum) + beta*C(i,j);
                }
            }
        }
    }
}

#ifdef TLAPACK_USE_BLAS

// Wrapper for BLAS types

template<
    class matrixA_t,
    class matrixB_t, 
    class matrixC_t, 
    class alpha_t, 
    class beta_t,
    _ENABLE_IF_HAS_BLAS_TYPE(matrixA_t, matrixB_t, matrixC_t, alpha_t, beta_t) = 0
>
inline
void gemm(
    Op transA,
    Op transB,
    const alpha_t& alpha,
    const matrixA_t& A,
    const matrixB_t& B,
    const beta_t& beta,
    matrixC_t& C )
{
    // Legacy objects
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);
    auto _C = legacy_matrix(C);

    // Constants to forward
    auto&& m = _C.m;
    auto&& n = _C.n;
    auto&& k = (transA == Op::NoTrans) ? _A.n : _A.m;

    gemm(
        _A.layout, transA, transB, 
        m, n, k,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim,
        beta,
        _C.ptr, _C.ldim );
}

#endif // TLAPACK_USE_BLAS

}  // namespace blas

#endif        //  #ifndef BLAS_GEMM_HH
