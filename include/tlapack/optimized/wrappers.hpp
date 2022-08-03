// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAPACKPP_WRAPPERS_HH
#define TLAPACK_LAPACKPP_WRAPPERS_HH

#include "lapack.hh" // from LAPACK++
#include "tlapack/base/utils.hpp"

namespace tlapack {

// =============================================================================
// Level 1 BLAS wrappers

template< class vector_t,
    enable_if_allow_optblas_t< vector_t > = 0
>
inline
auto asum( vector_t const& x )
{
    auto x_ = legacy_vector(x);
    return ::blas::asum( x_.n, x_.ptr, x_.inc );
}

template< class vectorX_t, class vectorY_t, class alpha_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void axpy(
    const alpha_t alpha,
    const vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = x_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::axpy( n, alpha, x_.ptr, incx, y_.ptr, incy );
}

template< class vectorX_t, class vectorY_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void copy( const vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = x_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::copy( n, x_.ptr, incx, y_.ptr, incy );
}

template< class vectorX_t, class vectorY_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
auto dot( const vectorX_t& x, const vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = x_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::dot( n, x_.ptr, incx, y_.ptr, incy );
}

template< class vectorX_t, class vectorY_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
auto dotu( const vectorX_t& x, const vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = x_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::dotu( n, x_.ptr, incx, y_.ptr, incy );
}

template< class vector_t,
    enable_if_allow_optblas_t< vector_t > = 0
>
inline
auto iamax( vector_t const& x )
{
    auto x_ = legacy_vector(x);
    return ::blas::iamax( x_.n, x_.ptr, x_.inc );
}

template< class vector_t,
    enable_if_allow_optblas_t< vector_t > = 0
>
inline
auto nrm2( vector_t const& x )
{
    auto x_ = legacy_vector(x);
    return ::blas::nrm2( x_.n, x_.ptr, x_.inc );
}

template<
    class vectorX_t, class vectorY_t,
    class c_type, class s_type,
    class T = type_t<vectorX_t>,
    enable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >,
        pair< c_type, real_type<T> >,
        pair< s_type, real_type<T> >
    > = 0
>
inline
void rot(
    vectorX_t& x, vectorY_t& y,
    const c_type c, const s_type s )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = x_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::rot( n, x_.ptr, incx, y_.ptr, incy, c, s );
}

template <typename T,
    enable_if_t< is_same_v< T, real_type<T> >, int > = 0,
    enable_if_allow_optblas_t< T > = 0
>
inline
void rotg( T& a, T& b, T& c, T& s )
{
    // Constants
    const T zero = 0;
    const T one  = 1;
    const T anorm = tlapack::abs(a);
    const T bnorm = tlapack::abs(b);

    T r;
    ::lapack::lartg( a, b, &c, &s, &r );
    
    // Return information on a and b:
    a = r;
    if( s == zero || c == zero || (anorm > bnorm) )
        b = s;
    else if ( c != zero )
        b = one / c;
    else
        b = one;
}

template <typename T,
    enable_if_t< !is_same_v< T, real_type<T> >, int > = 0,
    enable_if_allow_optblas_t< T > = 0
>
inline
void rotg( T& a, const T& b, real_type<T>& c, complex_type<T>& s )
{
    T r;
    ::lapack::lartg( a, b, &c, &s, &r );
    a = r;
}

template<
    int flag,
    class vectorX_t, class vectorY_t,
    enable_if_t<((-2 <= flag) && (flag <= 1)), int > = 0,
    class T = type_t<vectorX_t>,
    enable_if_t< is_same_v< T, real_type<T> >, int > = 0,
    enable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void rotm( vectorX_t& x, vectorY_t& y, const T h[4] )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = x_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;
    const T h_[] = { (T) flag, h[0], h[1], h[2], h[3] };

    return ::blas::rotm( n, x_.ptr, incx, y_.ptr, incy, h_ );
}

template< typename T,
    enable_if_t< is_same_v< T, real_type<T> >, int > = 0,
    enable_if_allow_optblas_t< T > = 0
>
inline
int rotmg( T& d1, T& d2, T& a, const T b, T h[4] )
{
    T param[5];
    ::blas::rotmg( &d1, &d2, &a, b, param );
    
    h[0] = param[1];
    h[1] = param[2];
    h[2] = param[3];
    h[3] = param[4];
    
    return param[0];
}

template< class vector_t, class alpha_t,
    class T = type_t<vector_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< vector_t, T >
    > = 0
>
inline
void scal( const alpha_t alpha, vector_t& x )
{
    auto x_ = legacy_vector(x);
    return ::blas::scal( x_.n, alpha, x_.ptr, x_.inc );
}

template< class vectorX_t, class vectorY_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void swap( vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = x_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::swap( n, x_.ptr, incx, y_.ptr, incy );
}

// =============================================================================
// Level 2 BLAS wrappers

/**
 * General matrix-vector multiply.
 * 
 * Wrapper to optimized BLAS.
 * 
 * @see gemv(
    Op trans,
    const alpha_t& alpha, const matrixA_t& A, const vectorX_t& x,
    const beta_t& beta, vectorY_t& y )
 * 
 * @ingroup gemv
 */
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
void gemv(
    Op trans,
    const alpha_t alpha, const matrixA_t& A, const vectorX_t& x,
    const beta_t beta, vectorY_t& y )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& m = A_.m;
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::gemv(
        (::blas::Layout) A_.layout,
        (::blas::Op) trans,
        m, n,
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
    class T = type_t<matrixA_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void ger(
    const alpha_t alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& m = A_.m;
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::ger(
        (::blas::Layout) A_.layout,
        m, n,
        alpha,
        x_.ptr, incx,
        y_.ptr, incy,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    class T = type_t<matrixA_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void geru(
    const alpha_t alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& m = A_.m;
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::geru(
        (::blas::Layout) A_.layout,
        m, n,
        alpha,
        x_.ptr, incx,
        y_.ptr, incy,
        A_.ptr, A_.ldim );
}

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
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

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
    class vectorX_t,
    class alpha_t,
    class T = type_t<matrixA_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, real_type<T> >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >
    > = 0
>
inline
void her(
    Uplo  uplo,
    const alpha_t alpha,
    const vectorX_t& x,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    
    return ::blas::her(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        x_.ptr, incx,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    class T = type_t<matrixA_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void her2(
    Uplo  uplo,
    const alpha_t alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::her2(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        x_.ptr, incx,
        y_.ptr, incy,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t, 
    class alpha_t, class beta_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >,
        pair< beta_t,    T >
    > = 0
>
inline
void symv(
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
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::symv(
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
    class vectorX_t,
    class alpha_t,
    class T = type_t<matrixA_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >
    > = 0
>
inline
void syr(
    Uplo  uplo,
    const alpha_t alpha,
    const vectorX_t& x,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    
    return ::blas::syr(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        x_.ptr, incx,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    class T = type_t<matrixA_t>,
    enable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
inline
void syr2(
    Uplo  uplo,
    const alpha_t alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    const idx_t incy = (y_.direction == Direction::Forward) ? y_.inc : -y_.inc;

    return ::blas::syr2(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        x_.ptr, incx,
        y_.ptr, incy,
        A_.ptr, A_.ldim );
}

template< class matrixA_t, class vectorX_t,
    class T = type_t<vectorX_t>,
    enable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< vectorX_t, T >
    > = 0
>
inline
void trmv(
    Uplo uplo,
    Op trans,
    Diag diag,
    const matrixA_t& A,
    vectorX_t& x )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    
    return ::blas::trmv(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        n,
        A_.ptr, A_.ldim,
        x_.ptr, incx );
}

template< class matrixA_t, class vectorX_t,
    class T = type_t<vectorX_t>,
    enable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< vectorX_t, T >
    > = 0
>
inline
void trsv(
    Uplo uplo,
    Op trans,
    Diag diag,
    const matrixA_t& A,
    vectorX_t& x )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto x_ = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (x_.direction == Direction::Forward) ? x_.inc : -x_.inc;
    
    return ::blas::trsv(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        n,
        A_.ptr, A_.ldim,
        x_.ptr, incx );
}

// =============================================================================
// Level 3 BLAS wrappers

/**
 * General matrix-matrix multiply.
 * 
 * Wrapper to optimized BLAS.
 * 
 * @see gemm(
    Op transA,
    Op transB,
    const alpha_t& alpha,
    const matrixA_t& A,
    const matrixB_t& B,
    const beta_t& beta,
    matrixC_t& C )
 * 
 * @ingroup gemm
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
void gemm(
    Op transA,
    Op transB,
    const alpha_t alpha,
    const matrixA_t& A,
    const matrixB_t& B,
    const beta_t beta,
    matrixC_t& C )
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto B_ = legacy_matrix(B);
    auto C_ = legacy_matrix(C);

    // Constants to forward
    const auto& m = C_.m;
    const auto& n = C_.n;
    const auto& k = (transA == Op::NoTrans) ? A_.n : A_.m;

    return ::blas::gemm(
        (::blas::Layout) A_.layout,
        (::blas::Op) transA, (::blas::Op) transB, 
        m, n, k,
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
    const beta_t& beta, matrixC_t& C )
 * 
 * @ingroup hemm
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
 * @ingroup herk
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
 * @see her2k(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t& beta, matrixC_t& C )
 * 
 * @ingroup her2k
 */
template<
    class matrixA_t, class matrixB_t, class matrixC_t, 
    class alpha_t, class beta_t,
    enable_if_t<(
    /* Requires: */
        ! is_complex<beta_t>::value
    ), int > = 0,
    class T  = type_t<matrixC_t>,
    enable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >,
        pair< beta_t,    real_type<T> >
    > = 0
>
inline
void her2k(
    Uplo uplo,
    Op trans,
    const alpha_t alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t beta, matrixC_t& C )
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto B_ = legacy_matrix(B);
    auto C_ = legacy_matrix(C);

    // Constants to forward
    const auto& n = C_.n;
    const auto& k = (trans == Op::NoTrans) ? A_.n : A_.m;

    return ::blas::her2k(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans, 
        n, k,
        alpha,
        A_.ptr, A_.ldim,
        B_.ptr, B_.ldim,
        beta,
        C_.ptr, C_.ldim );
}

/**
 * Symmetric matrix-matrix multiply.
 * 
 * Wrapper to optimized BLAS.
 * 
 * @see symm(
    Side side,
    Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t& beta, matrixC_t& C )
 * 
 * @ingroup symm
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
void symm(
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

    return ::blas::symm(
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
 * Symmetric rank-k update
 * 
 * Wrapper to optimized BLAS.
 * 
 * @see syrk(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A,
    const beta_t& beta, matrixC_t& C )
 * 
 * @ingroup syrk
 */
template<
    class matrixA_t, class matrixC_t, 
    class alpha_t, class beta_t,
    class T  = type_t<matrixC_t>,
    enable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >,
        pair< beta_t,    T >
    > = 0
>
inline
void syrk(
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

    return ::blas::syrk(
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
 * Symmetric rank-k update
 * 
 * Wrapper to optimized BLAS.
 * 
 * @see syr2k(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t& beta, matrixC_t& C )
 * 
 * @ingroup syr2k
 */
template<
    class matrixA_t, class matrixB_t, class matrixC_t, 
    class alpha_t, class beta_t,
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
void syr2k(
    Uplo uplo,
    Op trans,
    const alpha_t alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t beta, matrixC_t& C )
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto B_ = legacy_matrix(B);
    auto C_ = legacy_matrix(C);

    // Constants to forward
    const auto& n = C_.n;
    const auto& k = (trans == Op::NoTrans) ? A_.n : A_.m;

    return ::blas::syr2k(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans, 
        n, k,
        alpha,
        A_.ptr, A_.ldim,
        B_.ptr, B_.ldim,
        beta,
        C_.ptr, C_.ldim );
}

/**
 * Triangular matrix-matrix multiply.
 * 
 * Wrapper to optimized BLAS.
 * 
 * @see trmm(
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    const alpha_t alpha,
    const matrixA_t& A,
    matrixB_t& B )
 * 
 * @ingroup trmm
 */
template< class matrixA_t, class matrixB_t, class alpha_t,
    class T  = type_t<matrixB_t>,
    enable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< alpha_t,   T >
    > = 0
>
inline
void trmm(
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    const alpha_t alpha,
    const matrixA_t& A,
    matrixB_t& B )
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto B_ = legacy_matrix(B);

    // Constants to forward
    const auto& m = B_.m;
    const auto& n = B_.n;

    return ::blas::trmm(
        (::blas::Layout) A_.layout,
        (::blas::Side) side,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        m, n,
        alpha,
        A_.ptr, A_.ldim,
        B_.ptr, B_.ldim );
}

template< class matrixA_t, class matrixB_t, class alpha_t,
    class T  = type_t<matrixB_t>,
    enable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< alpha_t,   T >
    > = 0
>
inline
void trsm(
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    const alpha_t alpha,
    const matrixA_t& A,
    matrixB_t& B )
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto B_ = legacy_matrix(B);

    // Constants to forward
    const auto& m = B_.m;
    const auto& n = B_.n;

    return ::blas::trsm(
        (::blas::Layout) A_.layout,
        (::blas::Side) side,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        m, n,
        alpha,
        A_.ptr, A_.ldim,
        B_.ptr, B_.ldim );
}

}  // namespace tlapack

#endif // TLAPACK_LAPACKPP_WRAPPERS_HH
