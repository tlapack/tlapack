/// @file tlapack_EigenMatrixXf.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tlapack_EigenMatrixXf.hpp"

// clang-format off
#include <tlapack/plugins/eigen.hpp>
#include <tlapack.hpp>
// clang-format on

namespace impl = tlapack_eigenmatrixxf;

using tlapack::Diag;
using tlapack::Op;
using tlapack::Side;
using tlapack::Uplo;

// =============================================================================
// Level 1 BLAS template implementations

float impl::asum(const Eigen::VectorXf& x) { return tlapack::asum(x); }

void impl::axpy(float alpha, const Eigen::VectorXf& x, Eigen::VectorXf& y)
{
    tlapack::axpy(alpha, x, y);
}

void impl::copy(const Eigen::VectorXf& x, Eigen::VectorXf& y)
{
    tlapack::copy(x, y);
}

float impl::dot(const Eigen::VectorXf& x, const Eigen::VectorXf& y)
{
    return tlapack::dot(x, y);
}

float impl::dotu(const Eigen::VectorXf& x, const Eigen::VectorXf& y)
{
    return tlapack::dotu(x, y);
}

int impl::iamax(const Eigen::VectorXf& x) { return tlapack::iamax(x); }

float impl::nrm2(const Eigen::VectorXf& x) { return tlapack::nrm2(x); }

void impl::rot(Eigen::VectorXf& x, Eigen::VectorXf& y, float c, float s)
{
    tlapack::rot(x, y, c, s);
}

template <>
void impl::rotm<-1>(Eigen::VectorXf& x, Eigen::VectorXf& y, const float h[4])
{
    tlapack::rotm<-1>(x, y, h);
}
template <>
void impl::rotm<0>(Eigen::VectorXf& x, Eigen::VectorXf& y, const float h[4])
{
    tlapack::rotm<0>(x, y, h);
}
template <>
void impl::rotm<1>(Eigen::VectorXf& x, Eigen::VectorXf& y, const float h[4])
{
    tlapack::rotm<1>(x, y, h);
}
template <>
void impl::rotm<-2>(Eigen::VectorXf& x, Eigen::VectorXf& y, const float h[4])
{
    tlapack::rotm<-2>(x, y, h);
}

void impl::scal(float alpha, Eigen::VectorXf& x) { tlapack::scal(alpha, x); }

void impl::swap(Eigen::VectorXf& x, Eigen::VectorXf& y) { tlapack::swap(x, y); }

// =============================================================================
// Level 2 BLAS template implementations

void impl::gemv(Op trans,
                float alpha,
                const Eigen::MatrixXf& A,
                const Eigen::VectorXf& x,
                float beta,
                Eigen::VectorXf& y)
{
    tlapack::gemv(trans, alpha, A, x, beta, y);
}

void impl::ger(float alpha,
               const Eigen::VectorXf& x,
               const Eigen::VectorXf& y,
               Eigen::MatrixXf& A)
{
    tlapack::ger(alpha, x, y, A);
}

void impl::hemv(Uplo uplo,
                float alpha,
                const Eigen::MatrixXf& A,
                const Eigen::VectorXf& x,
                float beta,
                Eigen::VectorXf& y)
{
    tlapack::hemv(uplo, alpha, A, x, beta, y);
}

void impl::her(Uplo uplo,
               float alpha,
               const Eigen::VectorXf& x,
               Eigen::MatrixXf& A)
{
    tlapack::her(uplo, alpha, x, A);
}

void impl::her2(Uplo uplo,
                float alpha,
                const Eigen::VectorXf& x,
                const Eigen::VectorXf& y,
                Eigen::MatrixXf& A)
{
    tlapack::her2(uplo, alpha, x, y, A);
}

void impl::symv(Uplo uplo,
                float alpha,
                const Eigen::MatrixXf& A,
                const Eigen::VectorXf& x,
                float beta,
                Eigen::VectorXf& y)
{
    tlapack::symv(uplo, alpha, A, x, beta, y);
}

void impl::syr(Uplo uplo,
               float alpha,
               const Eigen::VectorXf& x,
               Eigen::MatrixXf& A)
{
    tlapack::syr(uplo, alpha, x, A);
}

void impl::syr2(Uplo uplo,
                float alpha,
                const Eigen::VectorXf& x,
                const Eigen::VectorXf& y,
                Eigen::MatrixXf& A)
{
    tlapack::syr2(uplo, alpha, x, y, A);
}

void impl::trmv(Uplo uplo,
                Op trans,
                Diag diag,
                const Eigen::MatrixXf& A,
                Eigen::VectorXf& x)
{
    tlapack::trmv(uplo, trans, diag, A, x);
}

void impl::trsv(Uplo uplo,
                Op trans,
                Diag diag,
                const Eigen::MatrixXf& A,
                Eigen::VectorXf& x)
{
    tlapack::trsv(uplo, trans, diag, A, x);
}

// =============================================================================
// Level 3 BLAS template implementations

void impl::gemm(Op transA,
                Op transB,
                float alpha,
                const Eigen::MatrixXf& A,
                const Eigen::MatrixXf& B,
                float beta,
                Eigen::MatrixXf& C)
{
    tlapack::gemm(transA, transB, alpha, A, B, beta, C);
}

void impl::hemm(Side side,
                Uplo uplo,
                float alpha,
                const Eigen::MatrixXf& A,
                const Eigen::MatrixXf& B,
                float beta,
                Eigen::MatrixXf& C)
{
    tlapack::hemm(side, uplo, alpha, A, B, beta, C);
}

void impl::herk(Uplo uplo,
                Op trans,
                float alpha,
                const Eigen::MatrixXf& A,
                float beta,
                Eigen::MatrixXf& C)
{
    tlapack::herk(uplo, trans, alpha, A, beta, C);
}

void impl::her2k(Uplo uplo,
                 Op trans,
                 float alpha,
                 const Eigen::MatrixXf& A,
                 const Eigen::MatrixXf& B,
                 float beta,
                 Eigen::MatrixXf& C)
{
    tlapack::her2k(uplo, trans, alpha, A, B, beta, C);
}

void impl::symm(Side side,
                Uplo uplo,
                float alpha,
                const Eigen::MatrixXf& A,
                const Eigen::MatrixXf& B,
                float beta,
                Eigen::MatrixXf& C)
{
    tlapack::symm(side, uplo, alpha, A, B, beta, C);
}

void impl::syrk(Uplo uplo,
                Op trans,
                float alpha,
                const Eigen::MatrixXf& A,
                float beta,
                Eigen::MatrixXf& C)
{
    tlapack::syrk(uplo, trans, alpha, A, beta, C);
}

void impl::syr2k(Uplo uplo,
                 Op trans,
                 float alpha,
                 const Eigen::MatrixXf& A,
                 const Eigen::MatrixXf& B,
                 float beta,
                 Eigen::MatrixXf& C)
{
    tlapack::syr2k(uplo, trans, alpha, A, B, beta, C);
}

void impl::trmm(Side side,
                Uplo uplo,
                Op trans,
                Diag diag,
                float alpha,
                const Eigen::MatrixXf& A,
                Eigen::MatrixXf& B)
{
    tlapack::trmm(side, uplo, trans, diag, alpha, A, B);
}

void impl::trsm(Side side,
                Uplo uplo,
                Op trans,
                Diag diag,
                float alpha,
                const Eigen::MatrixXf& A,
                Eigen::MatrixXf& B)
{
    tlapack::trsm(side, uplo, trans, diag, alpha, A, B);
}