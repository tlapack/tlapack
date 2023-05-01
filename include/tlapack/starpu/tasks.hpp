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
    void insert_task_potf2(uplo_t uplo,
                           starpu_data_handle_t A,
                           starpu_data_handle_t info)
    {
        using args_t = std::tuple<uplo_t>;

        // Allocate space for the task
        struct starpu_task* task = starpu_task_create();

        // Allocate space for the arguments
        args_t* args_ptr = new args_t;

        // Initialize arguments
        std::get<0>(*args_ptr) = uplo;

        // Initialize task
        task->cl = (struct starpu_codelet*)&(cl::potf2<uplo_t, T>);
        task->handles[0] = A;
        task->handles[1] = info;
        task->cl_arg = (void*)args_ptr;
        task->cl_arg_size = sizeof(args_t);
        task->callback_func = [](void* args) noexcept { delete (args_t*)args; };
        task->callback_arg = (void*)args_ptr;

        // Submit task
        const int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_TASKS_HH