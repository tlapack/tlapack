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

// <T>LAPACK headers
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/starpu/gemm.hpp>

// C++ headers
#include <iostream>

int main(int argc, char** argv)
{
    using namespace tlapack;
    using starpu::Matrix;
    using T = float;

    srand(3);
    T u = tlapack::uroundoff<T>();

    size_t m = 21;
    size_t n = 10;
    size_t k = 6;
    size_t r = 7;
    size_t s = 5;
    size_t t = 3;

    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) k = atoi(argv[3]);
    if (argc > 4) r = atoi(argv[4]);
    if (argc > 5) s = atoi(argv[5]);
    if (argc > 6) t = atoi(argv[6]);
    if (argc > 7 || (r > m) || (s > n) || (t > k) || (r == 0) || (s == 0) ||
        (t == 0)) {
        std::cout << "Usage: " << argv[0] << " [m] [n] [k] [r] [s]"
                  << std::endl;
        std::cout << "  m:      number of rows of C (default: 21)" << std::endl;
        std::cout << "  n:      number of columns of C (default: 10)"
                  << std::endl;
        std::cout << "  k:      number of columns of A and rows of B "
                     "(default: 6)"
                  << std::endl;
        std::cout << "  r:      number of tiles along the row dimension of C "
                     "(default: 7)."
                  << std::endl;
        std::cout << "  s:      number of tiles along the col dimension of C "
                     "(default: 5).";
        std::cout << "  t:      number of tiles along the col dimension of A "
                     "(default: 3)."
                  << std::endl;
        return 1;
    }

    // Print input parameters
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "k = " << k << std::endl;
    std::cout << "r = " << r << std::endl;
    std::cout << "s = " << s << std::endl;
    std::cout << "t = " << t << std::endl << std::endl;

    /* initialize StarPU */
    setenv("STARPU_CODELET_PROFILING", "0", 1);
    setenv("STARPU_SCHED", "dmdas", 1);
    setenv("HWLOC_COMPONENTS", "-gl", 1);
    const int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
    starpu_cublas_init();

    /* allocate data */
    T *A_, *B_, *C_;
    starpu_malloc((void**)&A_, m * k * sizeof(T));
    starpu_malloc((void**)&B_, k * n * sizeof(T));
    starpu_malloc((void**)&C_, m * n * sizeof(T));

    {
        /* create matrix A */
        for (size_t i = 0; i < m * k; i++) {
            A_[i] = T((float)rand() / RAND_MAX);
        }
        Matrix<T> A(A_, m, k, r, t);

        /* create matrix B */
        for (size_t i = 0; i < k * n; i++) {
            B_[i] = T((float)rand() / RAND_MAX);
        }
        Matrix<T> B(B_, k, n, t, s);

        /* create matrix C */
        for (size_t i = 0; i < m * n; i++) {
            C_[i] = T(0xdeadbeef);
        }
        Matrix<T> C(C_, m, n, r, s);

        /* GEMM */
        gemm(noTranspose, noTranspose, T(1), A, B, C);

        /* check the result */
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t l = 0; l < k; ++l)
                    C(i, j) -= A(i, l) * B(l, j);
    }

    T componentwise_relerror = 0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T aux = 0;
            for (size_t l = 0; l < k; ++l)
                aux += std::abs(A_[i + l * m] * B_[l + j * k]);
            componentwise_relerror =
                max(componentwise_relerror, std::abs(C_[i + j * m]) / aux);
        }
    }

    /* print the error */
    std::cout << "componentwise rel. error = " << componentwise_relerror
              << std::endl;
    std::cout << "gammak = " << (k * u) / (1 - k * u) << std::endl;

    /* free data */
    starpu_free_noflag(A_, m * k * sizeof(T));
    starpu_free_noflag(B_, k * n * sizeof(T));
    starpu_free_noflag(C_, m * n * sizeof(T));

    /* terminate StarPU */
    starpu_cublas_shutdown();
    starpu_shutdown();

    return 0;
}
