/// @file tlapack_EigenMatrixXf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EIGENMATRIXXF_HH
#define TLAPACK_EIGENMATRIXXF_HH

#include <Eigen/Dense>
#include <tlapack/base/types.hpp>

namespace tlapack_eigenmatrixxf {

using tlapack::Diag;
using tlapack::Op;
using tlapack::Side;
using tlapack::Uplo;

// =============================================================================
// Level 1 BLAS template implementations

float asum(const Eigen::VectorXf& x);
void axpy(float alpha, const Eigen::VectorXf& x, Eigen::VectorXf& y);
void copy(const Eigen::VectorXf& x, Eigen::VectorXf& y);
float dot(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
float dotu(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
int iamax(const Eigen::VectorXf& x);
float nrm2(const Eigen::VectorXf& x);
void rot(Eigen::VectorXf& x, Eigen::VectorXf& y, float c, float s);
template <int flag>
void rotm(Eigen::VectorXf& x, Eigen::VectorXf& y, const float h[4]);
void scal(float alpha, Eigen::VectorXf& x);
void swap(Eigen::VectorXf& x, Eigen::VectorXf& y);

// =============================================================================
// Level 2 BLAS template implementations

void gemv(Op trans,
          float alpha,
          const Eigen::MatrixXf& A,
          const Eigen::VectorXf& x,
          float beta,
          Eigen::VectorXf& y);
void ger(float alpha,
         const Eigen::VectorXf& x,
         const Eigen::VectorXf& y,
         Eigen::MatrixXf& A);
void geru(float alpha,
          const Eigen::VectorXf& x,
          const Eigen::VectorXf& y,
          Eigen::MatrixXf& A);
void hemv(Uplo uplo,
          float alpha,
          const Eigen::MatrixXf& A,
          const Eigen::VectorXf& x,
          float beta,
          Eigen::VectorXf& y);
void her(Uplo uplo, float alpha, const Eigen::VectorXf& x, Eigen::MatrixXf& A);
void her2(Uplo uplo,
          float alpha,
          const Eigen::VectorXf& x,
          const Eigen::VectorXf& y,
          Eigen::MatrixXf& A);
void symv(Uplo uplo,
          float alpha,
          const Eigen::MatrixXf& A,
          const Eigen::VectorXf& x,
          float beta,
          Eigen::VectorXf& y);
void syr(Uplo uplo, float alpha, const Eigen::VectorXf& x, Eigen::MatrixXf& A);
void syr2(Uplo uplo,
          float alpha,
          const Eigen::VectorXf& x,
          const Eigen::VectorXf& y,
          Eigen::MatrixXf& A);
void trmv(Uplo uplo,
          Op trans,
          Diag diag,
          const Eigen::MatrixXf& A,
          Eigen::VectorXf& x);
void trsv(Uplo uplo,
          Op trans,
          Diag diag,
          const Eigen::MatrixXf& A,
          Eigen::VectorXf& x);

// =============================================================================
// Level 3 BLAS template implementations

void gemm(Op transA,
          Op transB,
          float alpha,
          const Eigen::MatrixXf& A,
          const Eigen::MatrixXf& B,
          float beta,
          Eigen::MatrixXf& C);
void hemm(Side side,
          Uplo uplo,
          float alpha,
          const Eigen::MatrixXf& A,
          const Eigen::MatrixXf& B,
          float beta,
          Eigen::MatrixXf& C);
void herk(Uplo uplo,
          Op trans,
          float alpha,
          const Eigen::MatrixXf& A,
          float beta,
          Eigen::MatrixXf& C);
void her2k(Uplo uplo,
           Op trans,
           float alpha,
           const Eigen::MatrixXf& A,
           const Eigen::MatrixXf& B,
           float beta,
           Eigen::MatrixXf& C);
void symm(Side side,
          Uplo uplo,
          float alpha,
          const Eigen::MatrixXf& A,
          const Eigen::MatrixXf& B,
          float beta,
          Eigen::MatrixXf& C);
void syrk(Uplo uplo,
          Op trans,
          float alpha,
          const Eigen::MatrixXf& A,
          float beta,
          Eigen::MatrixXf& C);
void syr2k(Uplo uplo,
           Op trans,
           float alpha,
           const Eigen::MatrixXf& A,
           const Eigen::MatrixXf& B,
           float beta,
           Eigen::MatrixXf& C);
void trmm(Side side,
          Uplo uplo,
          Op trans,
          Diag diag,
          float alpha,
          const Eigen::MatrixXf& A,
          Eigen::MatrixXf& B);
void trsm(Side side,
          Uplo uplo,
          Op trans,
          Diag diag,
          float alpha,
          const Eigen::MatrixXf& A,
          Eigen::MatrixXf& B);

}  // namespace tlapack_eigenmatrixxf

#endif  // TLAPACK_EIGENMATRIXXF_HH
