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

// <T>LAPACK headers
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/starpu/starpu.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>

using tlapack::starpu::idx_t;

template <class T>
int run(idx_t n, idx_t nx, bool verbose = false)
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
    starpu_malloc((void**)&B_, n * n * sizeof(T));

    /* pin arrays to improve GPU/CPU transfers */
    starpu_memory_pin((void*)A_, n * n * sizeof(T));
    starpu_memory_pin((void*)B_, n * n * sizeof(T));

    /* A is symmetric positive definite and B is a copy of A */
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = 0; i < j; i++)
            A_[i + j * n] = A_[i * n + j] = T((float)rand() / RAND_MAX);
    for (idx_t i = 0; i < n; i++)
        A_[i + n * i] += T(i + 1);
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = 0; i < n; i++)
            B_[i * n + j] = A_[i * n + j];

    double elapsed_time;
    {
        /* create matrix A and B */
        Matrix<T> A(A_, n, n), B(B_, n, n);
        if (verbose) std::cout << "A = " << A << std::endl;

        /* potrf options */
        potrf_opts_t<idx_t> opts;
        opts.nb = n / nx;
        opts.variant = PotrfVariant::Blocked;

        /* create grids*/
        A.create_grid(nx, nx);
        B.create_grid(nx, nx);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        /* call potrf */
        int info = potrf(upperTriangle, A, opts);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        elapsed_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();

        if (info != 0) {
            std::cout << "Cholesky ended with info " << info << std::endl;
        }

        // Solve U^H U B = A_init
        trsm(left_side, upperTriangle, conjTranspose, nonUnit_diagonal, one, A,
             B);
        trsm(left_side, upperTriangle, noTranspose, nonUnit_diagonal, one, A,
             B);

        if (verbose) std::cout << "A = " << A << std::endl;
        if (verbose) std::cout << "B = " << B << std::endl;
    }

    // error = ||B-Id||_1 / ||Id||_1
    real_t error = 0;
    for (idx_t j = 0; j < n; ++j) {
        real_t loc_error = 0;
        for (idx_t i = 0; i < n; ++i)
            loc_error += std::abs(B_[i + j * n] - (i == j ? one : 0));
        error = std::max(error, loc_error);
    }
    error /= n;

    // Output
    std::cout << "U^H U R = A   =>   ||R-Id||_1 / ||Id||_1 = " << error
              << std::endl
              << "time = " << elapsed_time * 1.0e-9 << " s" << std::endl;

    // Clean up
    starpu_memory_unpin((void*)A_, n * n * sizeof(T));
    starpu_memory_unpin((void*)B_, n * n * sizeof(T));
    starpu_free_noflag(A_, n * n * sizeof(T));
    starpu_free_noflag(B_, n * n * sizeof(T));

    return 0;
}

int main(int argc, char** argv)
{
    // initialize random seed
    srand(3);

    idx_t n = 100;
    idx_t nx = 10;
    bool verbose = true;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) nx = atoi(argv[2]);
    if (argc > 3) verbose = (tolower(argv[3][0]) == 'v');

    if (argc > 4 || (n % nx != 0) || (nx > n)) {
        std::cout << "Usage: " << argv[0] << " [n] [nx] [verbose]" << std::endl;
        std::cout << "  n:      number of rows and columns of A (default: 50)"
                  << std::endl;
        std::cout << "  nx:      number of tiles in x and y directions of A "
                     "(default: 10)."
                  << std::endl;
        std::cout
            << "  verbose: print input and output matrices (default: true)"
            << std::endl;
        return 1;
    }

    // Print input parameters
    std::cout << "n = " << n << std::endl;
    std::cout << "nx = " << nx << std::endl << std::endl;

    /* initialize StarPU */
    const int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
    starpu_cublas_init();

    // Run tests:

    std::cout << "float:" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    if (run<float>(n, nx, verbose)) return 1;
    std::cout << std::endl;

    std::cout << "double:" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    if (run<double>(n, nx, verbose)) return 1;
    std::cout << std::endl;

    std::cout << "complex<float>:" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    if (run<std::complex<float>>(n, nx, verbose)) return 1;
    std::cout << std::endl;

    std::cout << "complex<double>:" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    if (run<std::complex<double>>(n, nx, verbose)) return 1;
    std::cout << std::endl;

    /* terminate StarPU */
    starpu_cublas_shutdown();
    starpu_shutdown();

    return 0;
}
