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
    alpha_t alpha,
    const matrixA_t& A,
    const matrixB_t& B,
    beta_t beta,
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

    ::blas::trsm(
        (::blas::Layout) _A.layout,
        (::blas::Side) side,
        (::blas::Uplo) uplo,
        (::blas::Op) trans,
        (::blas::Diag) diag,
        _B.m, _B.n,
        alpha,
        _A.ptr, _A.ldim,
        _B.ptr, _B.ldim );
}

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

}  // namespace tlapack

#endif // __TLAPACK_BLASPP_WRAPPERS_HH__