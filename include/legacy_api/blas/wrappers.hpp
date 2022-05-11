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
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& m = _C.m;
    const auto& n = _C.n;
    const auto& k = (transA == Op::NoTrans) ? _A.n : _A.m;

    ::blas::gemm(
        (::blas::Layout) _A.layout,
        (::blas::Op) transA, (::blas::Op) transB, 
        m, n, k,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim,
        beta,
        _C.ptr, _C.ldim );
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
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& m = _C.m;
    const auto& n = _C.n;

    ::blas::hemm(
        (::blas::Layout) _A.layout,
        (::blas::Side) side, (::blas::Uplo) uplo, 
        m, n,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim,
        beta,
        _C.ptr, _C.ldim );
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
    auto _A = legacy_matrix(A);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& n = _C.n;
    const auto& k = (trans == Op::NoTrans) ? _A.n : _A.m;

    ::blas::herk(
        (::blas::Layout) _A.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans, 
        n, k,
        alpha,
        _A.ptr, _A.ldim,
        beta,
        _C.ptr, _C.ldim );
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
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& n = _C.n;
    const auto& k = (trans == Op::NoTrans) ? _A.n : _A.m;

    ::blas::her2k(
        (::blas::Layout) _A.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans, 
        n, k,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim,
        beta,
        _C.ptr, _C.ldim );
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
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& m = _C.m;
    const auto& n = _C.n;

    ::blas::symm(
        (::blas::Layout) _A.layout,
        (::blas::Side) side, (::blas::Uplo) uplo, 
        m, n,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim,
        beta,
        _C.ptr, _C.ldim );
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
    auto _A = legacy_matrix(A);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& n = _C.n;
    const auto& k = (trans == Op::NoTrans) ? _A.n : _A.m;

    ::blas::syrk(
        (::blas::Layout) _A.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans, 
        n, k,
        alpha,
        _A.ptr, _A.ldim,
        beta,
        _C.ptr, _C.ldim );
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
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);
    auto _C = legacy_matrix(C);

    // Constants to forward
    const auto& n = _C.n;
    const auto& k = (trans == Op::NoTrans) ? _A.n : _A.m;

    ::blas::syr2k(
        (::blas::Layout) _A.layout,
        (::blas::Uplo) uplo,
        (::blas::Op) trans, 
        n, k,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim,
        beta,
        _C.ptr, _C.ldim );
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
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);

    // Constants to forward
    const auto& m = _B.m;
    const auto& n = _B.n;

    ::blas::trmm(
        (::blas::Layout) _A.layout,
        (::blas::Side) side,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        m, n,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim );
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
    auto _A = legacy_matrix(A);
    auto _B = legacy_matrix(B);

    // Constants to forward
    const auto& m = _B.m;
    const auto& n = _B.n;

    ::blas::trsm(
        (::blas::Layout) _A.layout,
        (::blas::Side) side,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        m, n,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim );
}

}  // namespace tlapack

#endif // __TLAPACK_BLASPP_WRAPPERS_HH__