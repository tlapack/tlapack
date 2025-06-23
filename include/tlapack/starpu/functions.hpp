/// @file starpu/functions.hpp
/// @brief StarPU functions for BLAS and LAPACK tasks.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_BLAS_CPU_HH
#define TLAPACK_STARPU_BLAS_CPU_HH

#include <starpu_cublas_v2.h>
#include <starpu_cusolver.h>

#include "tlapack/legacy_api/blas.hpp"
#include "tlapack/legacy_api/lapack/potrf.hpp"
#include "tlapack/starpu/utils.hpp"

namespace tlapack {
namespace starpu {
    namespace func {

        // ---------------------------------------------------------------------
        // Generic functions for BLAS routines

        template <class TA,
                  class TB,
                  class TC,
                  class alpha_t,
                  class beta_t,
                  int mode = 0>
        constexpr void gemm(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Op, Op, alpha_t, beta_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Op& transA = std::get<0>(cl_args);
            const Op& transB = std::get<1>(cl_args);
            const alpha_t& alpha = std::get<2>(cl_args);
            const beta_t& beta = std::get<3>(cl_args);

            // get dimensions
            const idx_t& m = STARPU_MATRIX_GET_NX(buffers[2]);
            const idx_t& n = STARPU_MATRIX_GET_NY(buffers[2]);
            const idx_t& k = (transA == Op::NoTrans)
                                 ? STARPU_MATRIX_GET_NY(buffers[0])
                                 : STARPU_MATRIX_GET_NX(buffers[0]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);
            const idx_t& ldc = STARPU_MATRIX_GET_LD(buffers[2]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_MATRIX_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_MATRIX_GET_PTR(buffers[2]);

            // call gemm
            if constexpr (mode == 0) {
                using T = scalar_type<TC, beta_t>;
                legacy::gemm(Layout::ColMajor, transA, transB, m, n, k, alpha,
                             (const TA*)A, lda, (const TB*)B, ldb, (T)beta,
                             (TC*)C, ldc);
            }
#ifdef STARPU_USE_CUDA
            else if constexpr (mode == 1) {
                using T = scalar_type<TA, TB, TC, alpha_t, beta_t>;

                const cublasOperation_t opA = cuda::op2cublas(transA);
                const cublasOperation_t opB = cuda::op2cublas(transB);
                const T alpha_ = (T)alpha;
                const T beta_ = (T)beta;
                cublasStatus_t status;

                if constexpr (is_same_v<T, float>)
                    status = cublasSgemm(starpu_cublas_get_local_handle(), opA,
                                         opB, m, n, k, &alpha_, (const float*)A,
                                         lda, (const float*)B, ldb, &beta_,
                                         (float*)C, ldc);
                else if constexpr (is_same_v<T, double>)
                    status = cublasDgemm(
                        starpu_cublas_get_local_handle(), opA, opB, m, n, k,
                        &alpha_, (const double*)A, lda, (const double*)B, ldb,
                        &beta_, (double*)C, ldc);
                else if constexpr (is_same_v<real_type<T>, float>)
                    status = cublasCgemm(
                        starpu_cublas_get_local_handle(), opA, opB, m, n, k,
                        (const cuFloatComplex*)&alpha_,
                        (const cuFloatComplex*)A, lda, (const cuFloatComplex*)B,
                        ldb, (const cuFloatComplex*)&beta_, (cuFloatComplex*)C,
                        ldc);
                else if constexpr (is_same_v<real_type<T>, double>)
                    status =
                        cublasZgemm(starpu_cublas_get_local_handle(), opA, opB,
                                    m, n, k, (const cuDoubleComplex*)&alpha_,
                                    (const cuDoubleComplex*)A, lda,
                                    (const cuDoubleComplex*)B, ldb,
                                    (const cuDoubleComplex*)&beta_,
                                    (cuDoubleComplex*)C, ldc);
                else
                    static_assert(sizeof(T) == 0,
                                  "Type not supported in cuBLAS");

                if (status != CUBLAS_STATUS_SUCCESS)
                    STARPU_CUBLAS_REPORT_ERROR(status);
            }
#endif
            else
                static_assert(mode == 0 || mode == 1, "Invalid mode");
        }

        template <class TA,
                  class TB,
                  class TC,
                  class alpha_t,
                  class beta_t,
                  int mode = 0>
        constexpr void symm(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Side, Uplo, alpha_t, beta_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Side& side = std::get<0>(cl_args);
            const Uplo& uplo = std::get<1>(cl_args);
            const alpha_t& alpha = std::get<2>(cl_args);
            const beta_t& beta = std::get<3>(cl_args);

            // get dimensions
            const idx_t& m = STARPU_MATRIX_GET_NX(buffers[2]);
            const idx_t& n = STARPU_MATRIX_GET_NY(buffers[2]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);
            const idx_t& ldc = STARPU_MATRIX_GET_LD(buffers[2]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_MATRIX_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_MATRIX_GET_PTR(buffers[2]);

            // call symm
            using T = scalar_type<TC, beta_t>;
            legacy::symm(Layout::ColMajor, side, uplo, m, n, alpha,
                         (const TA*)A, lda, (const TB*)B, ldb, (T)beta, (TC*)C,
                         ldc);
        }

        template <class TA,
                  class TB,
                  class TC,
                  class alpha_t,
                  class beta_t,
                  int mode = 0>
        constexpr void hemm(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Side, Uplo, alpha_t, beta_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Side& side = std::get<0>(cl_args);
            const Uplo& uplo = std::get<1>(cl_args);
            const alpha_t& alpha = std::get<2>(cl_args);
            const beta_t& beta = std::get<3>(cl_args);

            // get dimensions
            const idx_t& m = STARPU_MATRIX_GET_NX(buffers[2]);
            const idx_t& n = STARPU_MATRIX_GET_NY(buffers[2]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);
            const idx_t& ldc = STARPU_MATRIX_GET_LD(buffers[2]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_MATRIX_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_MATRIX_GET_PTR(buffers[2]);

            // call hemm
            using T = scalar_type<TC, beta_t>;
            legacy::hemm(Layout::ColMajor, side, uplo, m, n, alpha,
                         (const TA*)A, lda, (const TB*)B, ldb, (T)beta, (TC*)C,
                         ldc);
        }

        template <class TA, class TC, class alpha_t, class beta_t, int mode = 0>
        constexpr void syrk(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Uplo, Op, alpha_t, beta_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Uplo& uplo = std::get<0>(cl_args);
            const Op& op = std::get<1>(cl_args);
            const alpha_t& alpha = std::get<2>(cl_args);
            const beta_t& beta = std::get<3>(cl_args);

            // get dimensions
            const idx_t& n = STARPU_MATRIX_GET_NX(buffers[1]);
            const idx_t& k = (op == Op::NoTrans)
                                 ? STARPU_MATRIX_GET_NY(buffers[0])
                                 : STARPU_MATRIX_GET_NX(buffers[0]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldc = STARPU_MATRIX_GET_LD(buffers[1]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& C = STARPU_MATRIX_GET_PTR(buffers[1]);

            // call syrk
            using T = scalar_type<TC, beta_t>;
            legacy::syrk(Layout::ColMajor, uplo, op, n, k, alpha, (const TA*)A,
                         lda, (T)beta, (TC*)C, ldc);
        }

        template <class TA, class TC, class alpha_t, class beta_t, int mode = 0>
        constexpr void herk(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Uplo, Op, alpha_t, beta_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Uplo& uplo = std::get<0>(cl_args);
            const Op& op = std::get<1>(cl_args);
            const alpha_t& alpha = std::get<2>(cl_args);
            const beta_t& beta = std::get<3>(cl_args);

            // get dimensions
            const idx_t& n = STARPU_MATRIX_GET_NX(buffers[1]);
            const idx_t& k = (op == Op::NoTrans)
                                 ? STARPU_MATRIX_GET_NY(buffers[0])
                                 : STARPU_MATRIX_GET_NX(buffers[0]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldc = STARPU_MATRIX_GET_LD(buffers[1]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& C = STARPU_MATRIX_GET_PTR(buffers[1]);

            // call herk
            if constexpr (mode == 0) {
                using real_t = real_type<scalar_type<TC, beta_t>>;
                legacy::herk(Layout::ColMajor, uplo, op, n, k, alpha,
                             (const TA*)A, lda, (real_t)beta, (TC*)C, ldc);
            }
#ifdef STARPU_USE_CUDA
            else if constexpr (mode == 1) {
                using T = scalar_type<TA, TC, alpha_t, beta_t>;
                using real_t = real_type<T>;

                const cublasFillMode_t uplo_ = cuda::uplo2cublas(uplo);
                const cublasOperation_t op_ = cuda::op2cublas(op);
                const real_t alpha_ = (real_t)alpha;
                const real_t beta_ = (real_t)beta;
                cublasStatus_t status;

                if constexpr (is_same_v<T, float>)
                    status = cublasSsyrk(
                        starpu_cublas_get_local_handle(), uplo_, op_, n, k,
                        &alpha_, (const float*)A, lda, &beta_, (float*)C, ldc);
                else if constexpr (is_same_v<T, double>)
                    status =
                        cublasDsyrk(starpu_cublas_get_local_handle(), uplo_,
                                    op_, n, k, &alpha_, (const double*)A, lda,
                                    &beta_, (double*)C, ldc);
                else if constexpr (is_same_v<real_type<T>, float>)
                    status = cublasCherk(starpu_cublas_get_local_handle(),
                                         uplo_, op_, n, k, &alpha_,
                                         (const cuFloatComplex*)A, lda, &beta_,
                                         (cuFloatComplex*)C, ldc);
                else if constexpr (is_same_v<real_type<T>, double>)
                    status = cublasZherk(starpu_cublas_get_local_handle(),
                                         uplo_, op_, n, k, &alpha_,
                                         (const cuDoubleComplex*)A, lda, &beta_,
                                         (cuDoubleComplex*)C, ldc);
                else
                    static_assert(sizeof(T) == 0,
                                  "Type not supported in cuBLAS");

                if (status != CUBLAS_STATUS_SUCCESS)
                    STARPU_CUBLAS_REPORT_ERROR(status);
            }
#endif
            else
                static_assert(mode == 0 || mode == 1, "Invalid mode");
        }

        template <class TA,
                  class TB,
                  class TC,
                  class alpha_t,
                  class beta_t,
                  int mode = 0>
        constexpr void syr2k(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Uplo, Op, alpha_t, beta_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Uplo& uplo = std::get<0>(cl_args);
            const Op& op = std::get<1>(cl_args);
            const alpha_t& alpha = std::get<2>(cl_args);
            const beta_t& beta = std::get<3>(cl_args);

            // get dimensions
            const idx_t& n = STARPU_MATRIX_GET_NX(buffers[2]);
            const idx_t& k = (op == Op::NoTrans)
                                 ? STARPU_MATRIX_GET_NY(buffers[0])
                                 : STARPU_MATRIX_GET_NX(buffers[0]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);
            const idx_t& ldc = STARPU_MATRIX_GET_LD(buffers[2]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_MATRIX_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_MATRIX_GET_PTR(buffers[2]);

            // call syr2k
            using T = scalar_type<TC, beta_t>;
            legacy::syr2k(Layout::ColMajor, uplo, op, n, k, alpha, (const TA*)A,
                          lda, (const TB*)B, ldb, (T)beta, (TC*)C, ldc);
        }

        template <class TA,
                  class TB,
                  class TC,
                  class alpha_t,
                  class beta_t,
                  int mode = 0>
        constexpr void her2k(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Uplo, Op, alpha_t, beta_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Uplo& uplo = std::get<0>(cl_args);
            const Op& op = std::get<1>(cl_args);
            const alpha_t& alpha = std::get<2>(cl_args);
            const beta_t& beta = std::get<3>(cl_args);

            // get dimensions
            const idx_t& n = STARPU_MATRIX_GET_NX(buffers[2]);
            const idx_t& k = (op == Op::NoTrans)
                                 ? STARPU_MATRIX_GET_NY(buffers[0])
                                 : STARPU_MATRIX_GET_NX(buffers[0]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);
            const idx_t& ldc = STARPU_MATRIX_GET_LD(buffers[2]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_MATRIX_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_MATRIX_GET_PTR(buffers[2]);

            // call her2k
            using real_t = real_type<scalar_type<TC, beta_t>>;
            legacy::her2k(Layout::ColMajor, uplo, op, n, k, alpha, (const TA*)A,
                          lda, (const TB*)B, ldb, (real_t)beta, (TC*)C, ldc);
        }

        template <class TA, class TB, class alpha_t, int mode = 0>
        constexpr void trmm(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Side, Uplo, Op, Diag, alpha_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Side& side = std::get<0>(cl_args);
            const Uplo& uplo = std::get<1>(cl_args);
            const Op& op = std::get<2>(cl_args);
            const Diag& diag = std::get<3>(cl_args);
            const alpha_t& alpha = std::get<4>(cl_args);

            // get dimensions
            const idx_t& m = STARPU_MATRIX_GET_NX(buffers[1]);
            const idx_t& n = STARPU_MATRIX_GET_NY(buffers[1]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_MATRIX_GET_PTR(buffers[1]);

            // call trmm
            legacy::trmm(Layout::ColMajor, side, uplo, op, diag, m, n, alpha,
                         (const TA*)A, lda, (TB*)B, ldb);
        }

        template <class TA, class TB, class alpha_t, int mode = 0>
        constexpr void trsm(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Side, Uplo, Op, Diag, alpha_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const Side& side = std::get<0>(cl_args);
            const Uplo& uplo = std::get<1>(cl_args);
            const Op& op = std::get<2>(cl_args);
            const Diag& diag = std::get<3>(cl_args);
            const alpha_t& alpha = std::get<4>(cl_args);

            // get dimensions
            const idx_t& m = STARPU_MATRIX_GET_NX(buffers[1]);
            const idx_t& n = STARPU_MATRIX_GET_NY(buffers[1]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);

            // get matrices
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_MATRIX_GET_PTR(buffers[1]);

            // call trsm
            if constexpr (mode == 0)
                legacy::trsm(Layout::ColMajor, side, uplo, op, diag, m, n,
                             alpha, (const TA*)A, lda, (TB*)B, ldb);
#ifdef STARPU_USE_CUDA
            else if constexpr (mode == 1) {
                using T = scalar_type<TA, TB, alpha_t>;

                const cublasSideMode_t side_ = cuda::side2cublas(side);
                const cublasFillMode_t uplo_ = cuda::uplo2cublas(uplo);
                const cublasOperation_t op_ = cuda::op2cublas(op);
                const cublasDiagType_t diag_ = cuda::diag2cublas(diag);
                const T alpha_ = (T)alpha;
                cublasStatus_t status;

                if constexpr (is_same_v<T, float>)
                    status =
                        cublasStrsm(starpu_cublas_get_local_handle(), side_,
                                    uplo_, op_, diag_, m, n, &alpha_,
                                    (const float*)A, lda, (float*)B, ldb);
                else if constexpr (is_same_v<T, double>)
                    status =
                        cublasDtrsm(starpu_cublas_get_local_handle(), side_,
                                    uplo_, op_, diag_, m, n, &alpha_,
                                    (const double*)A, lda, (double*)B, ldb);
                else if constexpr (is_same_v<real_type<T>, float>)
                    status = cublasCtrsm(
                        starpu_cublas_get_local_handle(), side_, uplo_, op_,
                        diag_, m, n, (const cuFloatComplex*)&alpha_,
                        (const cuFloatComplex*)A, lda, (cuFloatComplex*)B, ldb);
                else if constexpr (is_same_v<real_type<T>, double>)
                    status = cublasZtrsm(starpu_cublas_get_local_handle(),
                                         side_, uplo_, op_, diag_, m, n,
                                         (const cuDoubleComplex*)&alpha_,
                                         (const cuDoubleComplex*)A, lda,
                                         (cuDoubleComplex*)B, ldb);
                else
                    static_assert(sizeof(T) == 0,
                                  "Type not supported in cuBLAS");

                if (status != CUBLAS_STATUS_SUCCESS)
                    STARPU_CUBLAS_REPORT_ERROR(status);
            }
#endif
            else
                static_assert(mode == 0 || mode == 1, "Invalid mode");
        }

        // ---------------------------------------------------------------------
        // Generic functions for LAPACK routines

        template <class uplo_t, class T, bool has_info, int mode = 0>
        constexpr void potrf(void** buffers, void* args)
        {
            using args_t = std::tuple<uplo_t>;

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const uplo_t& uplo = std::get<0>(cl_args);

            // get dimensions
            const idx_t& n = STARPU_MATRIX_GET_NX(buffers[0]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);

            // get matrix
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);

            // get info
            int* info = (has_info) ? (int*)STARPU_VARIABLE_GET_PTR(buffers[1])
                                   : (int*)nullptr;

            // call potrf
            if constexpr (mode == 0) {
                if constexpr (has_info)
                    *info = legacy::potrf(uplo, n, (T*)A, lda);
                else
                    legacy::potrf(uplo, n, (T*)A, lda);
            }
#ifdef STARPU_HAVE_LIBCUSOLVER
            if constexpr (mode == 1) {
                const uintptr_t& w =
                    STARPU_VARIABLE_GET_PTR(buffers[(has_info ? 2 : 1)]);
                const size_t& lwork =
                    STARPU_VARIABLE_GET_ELEMSIZE(buffers[(has_info ? 2 : 1)]);

                const cublasFillMode_t uplo_ = cuda::uplo2cublas(uplo);
                cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

                if constexpr (is_same_v<T, float>)
                    status = cusolverDnSpotrf(
                        starpu_cusolverDn_get_local_handle(), uplo_, n,
                        (float*)A, lda, (float*)w, lwork / sizeof(float), info);
                else if constexpr (is_same_v<T, double>)
                    status =
                        cusolverDnDpotrf(starpu_cusolverDn_get_local_handle(),
                                         uplo_, n, (double*)A, lda, (double*)w,
                                         lwork / sizeof(double), info);
                else if constexpr (is_same_v<real_type<T>, float>)
                    status = cusolverDnCpotrf(
                        starpu_cusolverDn_get_local_handle(), uplo_, n,
                        (cuFloatComplex*)A, lda, (cuFloatComplex*)w,
                        lwork / sizeof(cuFloatComplex), info);
                else if constexpr (is_same_v<real_type<T>, double>)
                    status = cusolverDnZpotrf(
                        starpu_cusolverDn_get_local_handle(), uplo_, n,
                        (cuDoubleComplex*)A, lda, (cuDoubleComplex*)w,
                        lwork / sizeof(cuDoubleComplex), info);
                else
                    static_assert(sizeof(T) == 0,
                                  "Type not supported in cuSolver");

                if (status != CUSOLVER_STATUS_SUCCESS)
                    STARPU_CUBLAS_REPORT_ERROR(status);
            }
#endif
            else
                static_assert(mode == 0, "Invalid mode");
        }

    }  // namespace func
}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_BLAS_CPU_HH
