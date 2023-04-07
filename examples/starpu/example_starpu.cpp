/// @file example_starpu.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/starpu.hpp>

// <T>LAPACK
#include <tlapack/base/constants.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf_level0.hpp>
#include <tlapack/lapack/lacpy.hpp>

// C++ headers
#include <iostream>

void cpu_func(void* buffers[], void* cl_arg) { printf("Hello world\n"); }
struct starpu_codelet cl = {.cpu_funcs = {cpu_func}, .nbuffers = 0};

int main(int argc, char** argv)
{
    using namespace tlapack;
    using namespace starpu;

    size_t m = 4;
    size_t n = 10;

    /* initialize StarPU */
    const int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    {
        /* create matrix A */
        float* A_;
        starpu_malloc((void**)&A_, m * n * sizeof(float));
        for (size_t i = 0; i < 40; i++) {
            A_[i] = i + 1;
        }
        Matrix<float> A(A_, 4, 10);
        // A.create_grid(2, 5); // Try to put a grid when this is working

        /* print matrix A */
        std::cout << "A = " << A << std::endl;
        assert(A.nrows() == 4);
        assert(A.ncols() == 10);
        assert(A(1, 2) == 10);
        assert(A(2, 5) == 23);

        /* Modify an entry of A*/
        A(1, 2) = float(0);
        A(3, 3) = A(0, 0);

        /* print matrix A */
        std::cout << "A = " << A << std::endl;
        assert(A(1, 2) < tlapack::ulp<float>());

        /* create a copy of matrix A in U */
        float* U_;
        starpu_malloc((void**)&U_, m * n * sizeof(float));
        Matrix<float> U(U_, 4, 10);
        lacpy(dense, A, U);

        /* LU factorization */
        std::vector<size_t> p(4);
        getrf_level0(U, p);

        /* Create and print matrix L */
        float* L_;
        starpu_malloc((void **)&L_, n*n*sizeof(float));
        Matrix<float> L(L_, 4, 4);
        lacpy(lowerTriangle, U, L);
        for (size_t i = 0; i < L.nrows(); ++i) {
            L(i, i) = 1;
        }
        std::cout << "L = " << L << std::endl;

        /* Create and print matrix U */
        for (size_t i = 0; i < U.nrows(); ++i)
            for (size_t j = 0; j < i; ++j)
                U(i, j) = 0;
        std::cout << "U = " << U << std::endl;

        /* print matrix L*U */
        trmm(left_side, lowerTriangle, noTranspose, unit_diagonal, 1, L, U);
        std::cout << "L*U = " << U << std::endl;

        /* Permute rows of A */
        for (size_t i = 0; i < A.nrows(); ++i) {
            if (p[i] != i) {
                for (size_t j = 0; j < A.ncols(); j++) {
                    float tmp = A(i, j);
                    A(i, j) = A(p[i], j);
                    A(p[i], j) = tmp;
                }
            }
        }

        /* Verify the factorization is good */
        for (size_t i = 0; i < A.nrows(); ++i) {
            for (size_t j = 0; j < A.ncols(); j++) {
                A(i, j) -= U(i, j);
            }
        }
        std::cout << "A - U = " << A << std::endl;

        // struct starpu_task* task = starpu_task_create();
        // task->cl = &cl; /* Pointer to the codelet defined above */
        // /* starpu_task_submit will be a blocking call. If unset,
        // starpu_task_wait() needs to be called after submitting the task.
        // */
        // task->synchronous = 1;
        // /* submit the task to StarPU */
        // starpu_task_submit(task);

        starpu_free_noflag(L_, n*n*sizeof(int));
        starpu_free_noflag(U_, m*n*sizeof(int));
        starpu_free_noflag(A_, m * n * sizeof(int));
    }

    /* terminate StarPU */
    starpu_shutdown();

    return 0;
}
