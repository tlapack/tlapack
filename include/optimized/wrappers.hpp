// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLASPP_WRAPPERS_HH__
#define __TLAPACK_BLASPP_WRAPPERS_HH__

#include "blas/wrappers.hh" // from BLAS++
#include "base/utils.hpp"

namespace tlapack {

// =============================================================================
// Level 1 BLAS wrappers

template< class vector_t,
    enable_if_allow_optblas_t< vector_t > = 0
>
inline
auto asum( vector_t const& x )
{
    auto _x = legacy_vector(x);
    return ::blas::asum( _x.n, _x.ptr, _x.inc );
}

template< class vectorX_t, class vectorY_t, class alpha_t,
    enable_if_allow_optblas_t<
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >
    > = 0
>
inline
void axpy(
    const alpha_t alpha,
    const vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = _x.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::axpy( n, alpha, _x.ptr, incx, _y.ptr, incy );
}

template< class vectorX_t, class vectorY_t,
    enable_if_allow_optblas_t<
        pair< vectorX_t, type_t< vectorX_t > >,
        pair< vectorY_t, type_t< vectorX_t > >
    > = 0
>
inline
void copy( const vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = _x.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::copy( n, _x.ptr, incx, _y.ptr, incy );
}

template< class vectorX_t, class vectorY_t,
    enable_if_allow_optblas_t<
        pair< vectorX_t, type_t< vectorX_t > >,
        pair< vectorY_t, type_t< vectorX_t > >
    > = 0
>
inline
auto dot( const vectorX_t& x, const vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = _x.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::dot( n, _x.ptr, incx, _y.ptr, incy );
}

template< class vectorX_t, class vectorY_t,
    enable_if_allow_optblas_t<
        pair< vectorX_t, type_t< vectorX_t > >,
        pair< vectorY_t, type_t< vectorX_t > >
    > = 0
>
inline
auto dotu( const vectorX_t& x, const vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = _x.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::dotu( n, _x.ptr, incx, _y.ptr, incy );
}

template< class vector_t,
    enable_if_allow_optblas_t< vector_t > = 0
>
inline
auto iamax( vector_t const& x )
{
    auto _x = legacy_vector(x);
    return ::blas::iamax( _x.n, _x.ptr, _x.inc );
}

template< class vector_t,
    enable_if_allow_optblas_t< vector_t > = 0
>
inline
auto nrm2( vector_t const& x )
{
    auto _x = legacy_vector(x);
    return ::blas::nrm2( _x.n, _x.ptr, _x.inc );
}

template<
    class vectorX_t, class vectorY_t,
    class c_type, class s_type,
    class T = vectorX_t,
    class real_t = real_type< T >,
    enable_if_allow_optblas_t<
        pair< vectorX_t, T >,
        pair< vectorY_t, T >,
        pair< c_type, real_t >,
        pair< s_type, real_t >
    > = 0
>
inline
void rot(
    vectorX_t& x, vectorY_t& y,
    const c_type c, const s_type s )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = _x.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::rot( n, _x.ptr, incx, _y.ptr, incy, c, s );
}

template <typename T,
    enable_if_allow_optblas_t< T > = 0
>
inline
void rotg( T& a, const T& b, real_type<T>& c, T& s )
{
    return ::blas::rotg( &a, (T*) &b, &c, &s );
}

template<
    int flag,
    class vectorX_t, class vectorY_t, class real_t,
    enable_if_t<((-2 <= flag) && (flag <= 1)), int > = 0,
    enable_if_allow_optblas_t<
        pair< real_t, real_type<real_t> >,
        pair< vectorX_t, real_t >,
        pair< vectorY_t, real_t >
    > = 0
>
inline
void rotm( vectorX_t& x, vectorY_t& y, const real_t h[4] )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = _x.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;
    const real_t _h[] = { (real_t) flag, h[0], h[1], h[2], h[3] };

    return ::blas::rotm( n, _x.ptr, incx, _y.ptr, incy, _h );
}

template< typename real_t,
    enable_if_allow_optblas_t<
        pair< real_t, real_type<real_t> >
    > = 0
>
inline
int rotmg(
    real_t& d1, real_t& d2,
    real_t& a, const real_t b,
    real_t h[4] )
{
    real_t param[5];
    ::blas::rotmg( &d1, &d2, &a, b, param );
    
    h[0] = param[1];
    h[1] = param[2];
    h[2] = param[3];
    h[3] = param[4];
    
    return param[0];
}

template< class vector_t, class alpha_t,
    enable_if_allow_optblas_t<
        pair< vector_t, alpha_t >
    > = 0
>
inline
void scal( const alpha_t alpha, vector_t& x )
{
    auto _x = legacy_vector(x);
    return ::blas::scal( _x.n, alpha, _x.ptr, _x.inc );
}

template< class vectorX_t, class vectorY_t,
    enable_if_allow_optblas_t<
        pair< vectorX_t, type_t< vectorX_t > >,
        pair< vectorY_t, type_t< vectorX_t > >
    > = 0
>
inline
void swap( vectorX_t& x, vectorY_t& y )
{
    using idx_t = size_type< vectorX_t >;

    // Legacy objects
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = _x.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::swap( n, _x.ptr, incx, _y.ptr, incy );
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
    enable_if_allow_optblas_t<
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >,
        pair< beta_t,    alpha_t >
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
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& m = A_.m;
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::gemv(
        (::blas::Layout) A_.layout,
        (::blas::Op) trans,
        m, n,
        alpha,
        A_.ptr, A_.ldim,
        _x.ptr, incx,
        beta,
        _y.ptr, incy );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >
    > = 0
>
inline
void ger(
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& m = A_.m;
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::ger(
        (::blas::Layout) A_.layout,
        m, n,
        alpha,
        _x.ptr, incx,
        _y.ptr, incy,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >
    > = 0
>
inline
void geru(
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& m = A_.m;
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::geru(
        (::blas::Layout) A_.layout,
        m, n,
        alpha,
        _x.ptr, incx,
        _y.ptr, incy,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t, 
    class alpha_t, class beta_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >,
        pair< beta_t,    alpha_t >
    > = 0
>
inline
void hemv(
    Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const vectorX_t& x,
    const beta_t& beta, vectorY_t& y )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::hemv(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        A_.ptr, A_.ldim,
        _x.ptr, incx,
        beta,
        _y.ptr, incy );
}

template<
    class matrixA_t,
    class vectorX_t,
    class alpha_t,
    enable_if_allow_optblas_t<
        pair< alpha_t, real_type<type_t<matrixA_t>> >,
        pair< matrixA_t, type_t<matrixA_t> >,
        pair< vectorX_t, type_t<matrixA_t> >
    > = 0
>
inline
void her(
    Uplo  uplo,
    const alpha_t& alpha,
    const vectorX_t& x,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    
    return ::blas::her(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        _x.ptr, incx,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >
    > = 0
>
inline
void her2(
    Uplo  uplo,
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::her2(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        _x.ptr, incx,
        _y.ptr, incy,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t, 
    class alpha_t, class beta_t,
    enable_if_allow_optblas_t<
        pair< alpha_t, real_type<alpha_t> >,
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >,
        pair< beta_t,    alpha_t >
    > = 0
>
inline
void symv(
    Uplo uplo,
    const alpha_t& alpha, const matrixA_t& A, const vectorX_t& x,
    const beta_t& beta, vectorY_t& y )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::symv(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        A_.ptr, A_.ldim,
        _x.ptr, incx,
        beta,
        _y.ptr, incy );
}

template<
    class matrixA_t,
    class vectorX_t,
    class alpha_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >
    > = 0
>
inline
void syr(
    Uplo  uplo,
    const alpha_t& alpha,
    const vectorX_t& x,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    
    return ::blas::syr(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        _x.ptr, incx,
        A_.ptr, A_.ldim );
}

template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, alpha_t >,
        pair< vectorX_t, alpha_t >,
        pair< vectorY_t, alpha_t >
    > = 0
>
inline
void syr2(
    Uplo  uplo,
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    using idx_t = size_type< matrixA_t >;

    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto _x = legacy_vector(x);
    auto _y = legacy_vector(y);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    const idx_t incy = (_y.direction == Direction::Forward) ? _y.inc : -_y.inc;

    return ::blas::syr2(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        n,
        alpha,
        _x.ptr, incx,
        _y.ptr, incy,
        A_.ptr, A_.ldim );
}

template< class matrixA_t, class vectorX_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, type_t< matrixA_t > >,
        pair< vectorX_t, type_t< matrixA_t > >
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
    auto _x = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    
    return ::blas::trmv(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        n,
        A_.ptr, A_.ldim,
        _x.ptr, incx );
}

template< class matrixA_t, class vectorX_t,
    enable_if_allow_optblas_t<
        pair< matrixA_t, type_t< matrixA_t > >,
        pair< vectorX_t, type_t< matrixA_t > >
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
    auto _x = legacy_vector(x);

    // Constants to forward
    const idx_t& n = A_.n;
    const idx_t incx = (_x.direction == Direction::Forward) ? _x.inc : -_x.inc;
    
    return ::blas::trsv(
        (::blas::Layout) A_.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        n,
        A_.ptr, A_.ldim,
        _x.ptr, incx );
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
    class T  = alpha_t,
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
    class T  = alpha_t,
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
    class T  = type_t<matrixA_t>,
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
    class T  = type_t<matrixA_t>,
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
    class T  = alpha_t,
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
    class T  = alpha_t,
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
    class T  = alpha_t,
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
    class T  = alpha_t,
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
    class T  = alpha_t,
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

#endif // __TLAPACK_BLASPP_WRAPPERS_HH__