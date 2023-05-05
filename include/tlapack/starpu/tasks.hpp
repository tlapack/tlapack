/// @file starpu/tasks.hpp
/// @brief Task insertion functions.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_TASKS_HH
#define TLAPACK_STARPU_TASKS_HH

#include "tlapack/starpu/codelets.hpp"

namespace tlapack {
namespace starpu {

    template <class TA, class TB, class TC, class alpha_t, class beta_t>
    void insert_task_gemm(Op transA,
                          Op transB,
                          const alpha_t& alpha,
                          starpu_data_handle_t A,
                          starpu_data_handle_t B,
                          const beta_t& beta,
                          starpu_data_handle_t C)
    {
        using args_t = std::tuple<Op, Op, alpha_t, beta_t>;

        // Allocate space for the task
        struct starpu_task* task = starpu_task_create();

        // Allocate space for the arguments
        args_t* args_ptr = new args_t;

        // Initialize arguments
        std::get<0>(*args_ptr) = transA;
        std::get<1>(*args_ptr) = transB;
        std::get<2>(*args_ptr) = alpha;
        std::get<3>(*args_ptr) = beta;

        // Initialize task
        task->cl =
            (struct starpu_codelet*)&(cl::gemm<TA, TB, TC, alpha_t, beta_t>);
        task->handles[0] = A;
        task->handles[1] = B;
        task->handles[2] = C;
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }

    template <class TA, class TC, class alpha_t, class beta_t>
    void insert_task_herk(Uplo uplo,
                          Op trans,
                          const alpha_t& alpha,
                          starpu_data_handle_t A,
                          const beta_t& beta,
                          starpu_data_handle_t C)
    {
        using args_t = std::tuple<Uplo, Op, alpha_t, beta_t>;

        // Allocate space for the task
        struct starpu_task* task = starpu_task_create();

        // Allocate space for the arguments
        args_t* args_ptr = new args_t;

        // Initialize arguments
        std::get<0>(*args_ptr) = uplo;
        std::get<1>(*args_ptr) = trans;
        std::get<2>(*args_ptr) = alpha;
        std::get<3>(*args_ptr) = beta;

        // Initialize task
        task->cl = (struct starpu_codelet*)&(cl::herk<TA, TC, alpha_t, beta_t>);
        task->handles[0] = A;
        task->handles[1] = C;
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }

    template <class TA, class TB, class alpha_t>
    void insert_task_trsm(Side side,
                          Uplo uplo,
                          Op trans,
                          Diag diag,
                          const alpha_t& alpha,
                          starpu_data_handle_t A,
                          starpu_data_handle_t B)
    {
        using args_t = std::tuple<Side, Uplo, Op, Diag, alpha_t>;

        // Allocate space for the task
        struct starpu_task* task = starpu_task_create();

        // Allocate space for the arguments
        args_t* args_ptr = new args_t;

        // Initialize arguments
        std::get<0>(*args_ptr) = side;
        std::get<1>(*args_ptr) = uplo;
        std::get<2>(*args_ptr) = trans;
        std::get<3>(*args_ptr) = diag;
        std::get<4>(*args_ptr) = alpha;

        // Initialize task
        task->cl = (struct starpu_codelet*)&(cl::trsm<TA, TB, alpha_t>);
        task->handles[0] = A;
        task->handles[1] = B;
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }

    template <class uplo_t, class T>
    void insert_task_potrf(uplo_t uplo,
                           starpu_data_handle_t A,
                           starpu_data_handle_t info = nullptr)
    {
        using args_t = std::tuple<uplo_t>;
        constexpr bool use_cusolver = cuda::is_cusolver_v<T>;

        // constants
        const bool has_info = (info != nullptr);

        // Allocate space for the task
        struct starpu_task* task = starpu_task_create();

        // Allocate space for the arguments
        args_t* args_ptr = new args_t;

        // Initialize arguments
        std::get<0>(*args_ptr) = uplo;

        // Initialize task
        task->cl = (struct starpu_codelet*)&(
            has_info ? cl::potrf<uplo_t, T> : cl::potrf_noinfo<uplo_t, T>);
        task->handles[0] = A;
        if (has_info) task->handles[1] = info;
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;

#ifdef STARPU_HAVE_LIBCUSOLVER
        starpu_data_handle_t& work = task->handles[(has_info ? 2 : 1)];
        int lwork = 0;
        if (use_cusolver && starpu_cuda_worker_get_count() > 0) {
            const cublasFillMode_t uplo_ = cuda::uplo2cublas(uplo);
            const int n = starpu_matrix_get_nx(A);

            if constexpr (std::is_same_v<T, float>) {
                cusolverDnSpotrf_bufferSize(
                    starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr, n,
                    &lwork);
                lwork *= sizeof(float);
            }
            else if constexpr (std::is_same_v<T, double>) {
                cusolverDnDpotrf_bufferSize(
                    starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr, n,
                    &lwork);
                lwork *= sizeof(double);
            }
            else if constexpr (std::is_same_v<real_type<T>, float>) {
                cusolverDnCpotrf_bufferSize(
                    starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr, n,
                    &lwork);
                lwork *= sizeof(cuFloatComplex);
            }
            else if constexpr (std::is_same_v<real_type<T>, double>) {
                cusolverDnZpotrf_bufferSize(
                    starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr, n,
                    &lwork);
                lwork *= sizeof(cuDoubleComplex);
            }
            else
                static_assert(sizeof(T) == 0, "Type not supported in cuSolver");
        }
        starpu_variable_data_register(&work, -1, 0, lwork);
#endif

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

#ifdef STARPU_HAVE_LIBCUSOLVER
        if constexpr (use_cusolver) starpu_data_unregister_submit(work);
#endif
    }

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_TASKS_HH
