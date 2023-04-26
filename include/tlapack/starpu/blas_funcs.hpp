/// @file starpu/blas_cpu.hpp
/// @brief BLAS Level 3 routines for StarPU (CPU backend)
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_BLAS_CPU_HH
#define TLAPACK_STARPU_BLAS_CPU_HH

#include <starpu_cublas_v2.h>

#include "tlapack/cuda/utils.hpp"
#include "tlapack/legacy_api/blas.hpp"

namespace tlapack {
namespace starpu {

#ifdef STARPU_USE_CUDA
    namespace cuda {

        inline cublasOperation_t op2cublas(Op op)
        {
            switch (op) {
                case Op::NoTrans:
                    return CUBLAS_OP_N;
                case Op::Trans:
                    return CUBLAS_OP_T;
                case Op::ConjTrans:
                    return CUBLAS_OP_C;
                case Op::Conj:
                    return CUBLAS_OP_CONJG;
                default:
                    throw std::invalid_argument("Invalid value for Op");
            }
        }

    }  // namespace cuda
#endif

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
            using T = scalar_type<TA, TB, TC, alpha_t, beta_t>;

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
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_VARIABLE_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_VARIABLE_GET_PTR(buffers[2]);

            // call gemm
            if constexpr (mode == 0)
                gemm(Layout::ColMajor, transA, transB, m, n, k, alpha,
                     (const TA*)A, lda, (const TB*)B, ldb, (T)beta, (TC*)C,
                     ldc);
            else if constexpr (mode == 1) {
                const cublasOperation_t opA = cuda::op2cublas(transA);
                const cublasOperation_t opB = cuda::op2cublas(transB);
                const T alpha_ = (T)alpha;
                const T beta_ = (T)beta;
                cublasStatus_t status;

                if constexpr (std::is_same_v<T, float>)
                    status =
                        cublasSgemm(starpu_cublas_get_local_handle(), opA, opB,
                                    m, n, k, &alpha_, (const T*)A, lda,
                                    (const T*)B, ldb, &beta_, (T*)C, ldc);
                else if constexpr (std::is_same_v<T, double>)
                    status =
                        cublasDgemm(starpu_cublas_get_local_handle(), opA, opB,
                                    m, n, k, &alpha_, (const T*)A, lda,
                                    (const T*)B, ldb, &beta_, (T*)C, ldc);
                else if constexpr (std::is_same_v<real_type<T>, float>)
                    status = cublasCgemm(
                        starpu_cublas_get_local_handle(), opA, opB, m, n, k,
                        (const cuComplex*)&alpha_, (const cuComplex*)A, lda,
                        (const cuComplex*)B, ldb, (const cuComplex*)&beta_,
                        (cuComplex*)C, ldc);
                else  // if constexpr (std::is_same_v<real_type<T>, double>)
                    status =
                        cublasZgemm(starpu_cublas_get_local_handle(), opA, opB,
                                    m, n, k, (const cuDoubleComplex*)&alpha_,
                                    (const cuDoubleComplex*)A, lda,
                                    (const cuDoubleComplex*)B, ldb,
                                    (const cuDoubleComplex*)&beta_,
                                    (cuDoubleComplex*)C, ldc);

                if (status != CUBLAS_STATUS_SUCCESS)
                    STARPU_CUBLAS_REPORT_ERROR(status);
            }
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
            using T = scalar_type<TA, TB, TC, alpha_t, beta_t>;

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
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_VARIABLE_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_VARIABLE_GET_PTR(buffers[2]);

            // call symm
            symm(Layout::ColMajor, side, uplo, m, n, alpha, (const TA*)A, lda,
                 (const TB*)B, ldb, (T)beta, (TC*)C, ldc);
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
            using T = scalar_type<TA, TB, TC, alpha_t, beta_t>;

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
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_VARIABLE_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_VARIABLE_GET_PTR(buffers[2]);

            // call hemm
            hemm(Layout::ColMajor, side, uplo, m, n, alpha, (const TA*)A, lda,
                 (const TB*)B, ldb, (T)beta, (TC*)C, ldc);
        }

        template <class TA, class TC, class alpha_t, class beta_t, int mode = 0>
        constexpr void syrk(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Uplo, Op, alpha_t, beta_t>;
            using T = scalar_type<TA, TC, alpha_t, beta_t>;

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
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& C = STARPU_VARIABLE_GET_PTR(buffers[1]);

            // call syrk
            syrk(Layout::ColMajor, uplo, op, n, k, alpha, (const TA*)A, lda,
                 (T)beta, (TC*)C, ldc);
        }

        template <class TA, class TC, class alpha_t, class beta_t, int mode = 0>
        constexpr void herk(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<Uplo, Op, alpha_t, beta_t>;
            using T = scalar_type<TA, TC, alpha_t, beta_t>;

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
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& C = STARPU_VARIABLE_GET_PTR(buffers[1]);

            // call herk
            herk(Layout::ColMajor, uplo, op, n, k, alpha, (const TA*)A, lda,
                 (T)beta, (TC*)C, ldc);
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
            using T = scalar_type<TA, TB, TC, alpha_t, beta_t>;

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
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_VARIABLE_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_VARIABLE_GET_PTR(buffers[2]);

            // call syr2k
            syr2k(Layout::ColMajor, uplo, op, n, k, alpha, (const TA*)A, lda,
                  (const TB*)B, ldb, (T)beta, (TC*)C, ldc);
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
            using T = scalar_type<TA, TB, TC, alpha_t, beta_t>;

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
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_VARIABLE_GET_PTR(buffers[1]);
            const uintptr_t& C = STARPU_VARIABLE_GET_PTR(buffers[2]);

            // call her2k
            her2k(Layout::ColMajor, uplo, op, n, k, alpha, (const TA*)A, lda,
                  (const TB*)B, ldb, (T)beta, (TC*)C, ldc);
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

            // get dimensions
            const idx_t& m = STARPU_MATRIX_GET_NX(buffers[1]);
            const idx_t& n = STARPU_MATRIX_GET_NY(buffers[1]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);

            // get matrices
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_VARIABLE_GET_PTR(buffers[1]);

            // call trmm
            trmm(Layout::ColMajor, side, uplo, op, diag, m, n, (const TA*)A,
                 lda, (TB*)B, ldb);
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

            // get dimensions
            const idx_t& m = STARPU_MATRIX_GET_NX(buffers[1]);
            const idx_t& n = STARPU_MATRIX_GET_NY(buffers[1]);
            const idx_t& lda = STARPU_MATRIX_GET_LD(buffers[0]);
            const idx_t& ldb = STARPU_MATRIX_GET_LD(buffers[1]);

            // get matrices
            const uintptr_t& A = STARPU_VARIABLE_GET_PTR(buffers[0]);
            const uintptr_t& B = STARPU_VARIABLE_GET_PTR(buffers[1]);

            // call trsm
            trsm(Layout::ColMajor, side, uplo, op, diag, m, n, (const TA*)A,
                 lda, (TB*)B, ldb);
        }

    }  // namespace internal

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_BLAS_CPU_HH