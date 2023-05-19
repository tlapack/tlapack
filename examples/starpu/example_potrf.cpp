/// @file examples/starpu/example_potrf.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/starpu.hpp>
#include <starpu_cublas_v2.h>
#include <starpu_cusolver.h>

// <T>LAPACK headers
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/starpu/starpu.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>

using tlapack::starpu::idx_t;

template <class T>
int run(idx_t n, idx_t nt, idx_t nb, bool check_error = false)
{
    using namespace tlapack;
    using starpu::Matrix;
    using tlapack::starpu::idx_t;
    using real_t = real_type<T>;

    // constant parameters
    const real_t one(1);

    /* create arrays A and B */
    T *A_, *B_;
    starpu_malloc((void**)&A_, n * n * sizeof(T));

    if (check_error) starpu_malloc((void**)&B_, n * n * sizeof(T));

    /* pin arrays to improve GPU/CPU transfers */
    starpu_memory_pin((void*)A_, n * n * sizeof(T));
    if (check_error) starpu_memory_pin((void*)B_, n * n * sizeof(T));

    /* A is symmetric positive definite and B is a copy of A */
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = 0; i < j; i++) {
            A_[i * n + j] = make_scalar<T>((float)rand() / (float)RAND_MAX,
                                           (float)rand() / (float)RAND_MAX);
            A_[i + j * n] = conj(A_[i * n + j]);
        }
    for (idx_t i = 0; i < n; i++)
        A_[i + n * i] = n + real_t((float)rand() / (float)RAND_MAX);
    if (check_error) {
        for (idx_t j = 0; j < n; j++)
            for (idx_t i = 0; i < n; i++)
                B_[i * n + j] = A_[i * n + j];
    }

    std::chrono::nanoseconds elapsed_time;
    {
        /* create matrix A and B */
        Matrix<T> A(A_, n, n, nt, nt);
        std::cout << "A = " << A << std::endl;

        /* potrf options */
        potrf_opts_t<idx_t> opts;
        opts.nb = nb;
        opts.variant = PotrfVariant::Blocked;

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        /* call potrf */
        int info = potrf(upperTriangle, A, opts);

        // Record end time
        starpu_task_wait_for_all();
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        elapsed_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        if (info != 0)
            std::cout << "Cholesky ended with info " << info << std::endl;

        if (check_error) {
            // Solve U^H U B = A_init
            Matrix<T> B(B_, n, n, nt, nt);
            trsm<Matrix<T>>(left_side, upperTriangle, conjTranspose,
                            nonUnit_diagonal, one, A, B);
            trsm<Matrix<T>>(left_side, upperTriangle, noTranspose,
                            nonUnit_diagonal, one, A, B);
            std::cout << "B = " << B << std::endl;
        }

        std::cout << "A = " << A << std::endl;
    }

    real_t error = 0;
    if (check_error) {
        // error = ||B-Id||_1 / ||Id||_1
        for (idx_t j = 0; j < n; ++j) {
            real_t loc_error = 0;
            for (idx_t i = 0; i < n; ++i)
                loc_error += std::abs(B_[i + j * n] - (i == j ? one : 0));
            error = std::max(error, loc_error);
        }
        error /= n;
    }

    // Output
    std::cout << "U^H U R = A   =>   ||R-Id||_1 / ||Id||_1 = "
              << ((check_error) ? error : real_t(-1)) << std::endl
              << "time = " << elapsed_time.count() * 1.0e-9 << " s"
              << std::endl;

    // Clean up
    starpu_memory_unpin((void*)A_, n * n * sizeof(T));
    if (check_error) starpu_memory_unpin((void*)B_, n * n * sizeof(T));
    starpu_free_noflag(A_, n * n * sizeof(T));
    if (check_error) starpu_free_noflag(B_, n * n * sizeof(T));

    return 0;
}

int main(int argc, char** argv)
{
    // initialize random seed
    srand(3);

    idx_t n = 50;
    idx_t nt = 30;
    idx_t nb = 28;
    int precision = 0b0001;
    bool check_error = true;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) nt = atoi(argv[2]);
    if (argc > 3) {
        char p = tolower(argv[3][0]);
        switch (p) {
            case 's':
                precision = 0b0001;
                break;
            case 'd':
                precision = 0b0010;
                break;
            case 'c':
                precision = 0b0100;
                break;
            case 'z':
                precision = 0b1000;
                break;
            case 'a':
                precision = 0b1111;
                break;
            default:
                precision = 0;
        }
    }
    if (argc > 4) check_error = (tolower(argv[4][0]) == 'y');
    if (argc > 5) nb = atoi(argv[5]);
    if (argc > 6 || (nt > n) || (n <= 0) || (nt <= 0) || precision == 0) {
        std::cout << "Usage: " << argv[0]
                  << " [n] [nt] [precision] [check_error]" << std::endl;
        std::cout << "  n:      number of rows and columns of A (default: 50)"
                  << std::endl;
        std::cout << "  nt:      number of rows and columns of tiles in A "
                     "(default: 10)."
                  << std::endl;
        std::cout << "  precision: s (0b0001), d (0b0010), c (0b0100), z "
                     "(0b1000), a (0b1111) (default: all (0b1111))."
                  << std::endl;
        std::cout << "  check_error: yes or no (default: no)" << std::endl;
        std::cout << "  nb:      Block size for the Cholesky factorization "
                     "(default: nt)."
                  << std::endl;
        return -1;
    }

    // Print input parameters
    std::cout << "n = " << n << std::endl;
    std::cout << "nt = " << nt << std::endl << std::endl;

    /* initialize StarPU */
    setenv("STARPU_CODELET_PROFILING", "0", 1);
    const int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
    starpu_cublas_init();
    starpu_cusolver_init();

    // Run tests:

    if (precision & 0b0001) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "float:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<float>(n, nt, nb, check_error)) return 1;
    }

    if (precision & 0b0010) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "double:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<double>(n, nt, nb, check_error)) return 2;
    }

    if (precision & 0b0100) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "complex<float>:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<std::complex<float>>(n, nt, nb, check_error)) return 3;
    }

    if (precision & 0b1000) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "complex<double>:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<std::complex<double>>(n, nt, nb, check_error)) return 4;
    }

    /* terminate StarPU */
    starpu_cusolver_shutdown();
    starpu_cublas_shutdown();
    starpu_shutdown();

    return 0;
}
