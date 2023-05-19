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
                          const Tile& A,
                          const Tile& B,
                          const beta_t& beta,
                          const Tile& C)
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

        // Handles
        starpu_data_handle_t handle[3];
        C.create_compatible_handles(handle, A, B);

        // Initialize task
        task->cl =
            (struct starpu_codelet*)&(cl::gemm<TA, TB, TC, alpha_t, beta_t>);
        task->handles[0] = handle[1];
        task->handles[1] = handle[2];
        task->handles[2] = handle[0];
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;
        // task->synchronous = 1;

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

        // Clean partition plan
        C.clean_compatible_handles(handle, A, B);
    }

    template <class TA, class TC, class alpha_t, class beta_t>
    void insert_task_herk(Uplo uplo,
                          Op trans,
                          const alpha_t& alpha,
                          const Tile& A,
                          const beta_t& beta,
                          const Tile& C)
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

        // Handles
        starpu_data_handle_t handle[2];
        A.create_compatible_handles(handle, C);

        // Initialize task
        task->cl = (struct starpu_codelet*)&(cl::herk<TA, TC, alpha_t, beta_t>);
        task->handles[0] = handle[0];
        task->handles[1] = handle[1];
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;
        // task->synchronous = 1;

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

        // Clean partition plan
        A.clean_compatible_handles(handle, C);
    }

    template <class TA, class TB, class alpha_t>
    void insert_task_trsm(Side side,
                          Uplo uplo,
                          Op trans,
                          Diag diag,
                          const alpha_t& alpha,
                          const Tile& A,
                          const Tile& B)
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

        // Handles
        starpu_data_handle_t handle[2];
        A.create_compatible_handles(handle, B);

        // Initialize task
        task->cl = (struct starpu_codelet*)&(cl::trsm<TA, TB, alpha_t>);
        task->handles[0] = handle[0];
        task->handles[1] = handle[1];
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;
        // task->synchronous = 1;

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

        // Clean partition plan
        A.clean_compatible_handles(handle, B);
    }

    template <class uplo_t, class T>
    void insert_task_potrf(uplo_t uplo,
                           const Tile& A,
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
        task->handles[0] = A.handle;
        if (has_info) task->handles[1] = info;
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;
        // task->synchronous = 1;

        if (use_cusolver) {
            int lwork = 0;
            if (starpu_cuda_worker_get_count() > 0) {
#ifdef STARPU_HAVE_LIBCUSOLVER
                const cublasFillMode_t uplo_ = cuda::uplo2cublas(uplo);
                const int n = starpu_matrix_get_nx(A.handle);

                if constexpr (std::is_same_v<T, float>) {
                    cusolverDnSpotrf_bufferSize(
                        starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr,
                        n, &lwork);
                    lwork *= sizeof(float);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    cusolverDnDpotrf_bufferSize(
                        starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr,
                        n, &lwork);
                    lwork *= sizeof(double);
                }
                else if constexpr (std::is_same_v<real_type<T>, float>) {
                    cusolverDnCpotrf_bufferSize(
                        starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr,
                        n, &lwork);
                    lwork *= sizeof(cuFloatComplex);
                }
                else if constexpr (std::is_same_v<real_type<T>, double>) {
                    cusolverDnZpotrf_bufferSize(
                        starpu_cusolverDn_get_local_handle(), uplo_, n, nullptr,
                        n, &lwork);
                    lwork *= sizeof(cuDoubleComplex);
                }
                else
                    static_assert(sizeof(T) == 0,
                                  "Type not supported in cuSolver");
#endif
            }
            starpu_variable_data_register(&(task->handles[(has_info ? 2 : 1)]),
                                          -1, 0, lwork);
        }

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

        if (use_cusolver)
            starpu_data_unregister_submit(task->handles[(has_info ? 2 : 1)]);
    }

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_TASKS_HH
