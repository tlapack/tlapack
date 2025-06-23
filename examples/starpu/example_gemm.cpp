/// @file example_starpu.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
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
    using range = pair<size_t, size_t>;

    srand(3);
    T u = tlapack::uroundoff<T>();

    size_t m = 4;
    size_t n = 4;
    size_t k = 3;
    size_t r = 4;
    size_t s = 4;
    size_t t = 2;

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
        std::cout << "  m:      number of rows of C (default: 4)" << std::endl;
        std::cout << "  n:      number of columns of C (default: 4)"
                  << std::endl;
        std::cout << "  k:      number of columns of A and rows of B "
                     "(default: 3)"
                  << std::endl;
        std::cout << "  r:      number of tiles along the row dimension of C "
                     "(default: 4)."
                  << std::endl;
        std::cout << "  s:      number of tiles along the col dimension of C "
                     "(default: 4).";
        std::cout << "  t:      number of tiles along the col dimension of A "
                     "(default: 2)."
                  << std::endl;
        return 1;
    }

    // for test purposes
    size_t d1 = 1;
    size_t d2 = 1;
    size_t d3 = 1;

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
    starpu_malloc((void**)&A_, (m + d1) * (k + d3) * sizeof(T));
    starpu_malloc((void**)&B_, (k + d3) * (n + d2) * sizeof(T));
    starpu_malloc((void**)&C_, (m + d1) * (n + d2) * sizeof(T));

    /* initialize data with garbage*/
    for (size_t i = 0; i < (m + d1) * (k + d3); i++)
        A_[i] = T(0xbeefdead);
    for (size_t i = 0; i < (k + d3) * (n + d2); i++)
        B_[i] = T(0xdeaddead);
    for (size_t i = 0; i < (m + d1) * (n + d2); i++)
        C_[i] = T(0xdeadbeef);

    {
        /* create matrix A */
        for (size_t j = 0; j < k; j++)
            for (size_t i = 0; i < m; i++)
                A_[(j + d3) * (m + d1) + (i + d1)] = 10 * i + j + 1;
        Matrix<T> Abig(A_, m + d1, k + d3, r, t);
        auto A = slice(Abig, range{d1, m + d1}, range{d3, k + d3});

        /* create matrix B */
        for (size_t j = 0; j < n; j++)
            for (size_t i = 0; i < k; i++)
                B_[(j + d2) * (k + d3) + (i + d3)] = 10 * i + j + 1;
        Matrix<T> Bbig(B_, k + d3, n + d2, t, s);
        auto B = slice(Bbig, range{d3, k + d3}, range{d2, n + d2});

        /* create matrix C */
        Matrix<T> Cbig(C_, m + d1, n + d2, r, s);
        auto C = slice(Cbig, range{d1, m + d1}, range{d2, n + d2});

        std::cout << "Abig = " << Abig << std::endl;
        std::cout << "A = " << A << std::endl;
        std::cout << "Bbig = " << Bbig << std::endl;
        std::cout << "B = " << B << std::endl;

        /* GEMM */
        gemm(NO_TRANS, NO_TRANS, T(1), A, B, C);

        std::cout << "A = " << A << std::endl;
        std::cout << "B = " << B << std::endl;
        std::cout << "C = " << C << std::endl;

        /* check the result */
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t l = 0; l < k; ++l)
                    C(i, j) -= A(i, l) * B(l, j);

        // std::cout << "C = " << C << std::endl;

        T componentwise_relerror = 0;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T aux = 0;
                for (size_t l = 0; l < k; ++l)
                    aux += std::abs(A(i, l) * B(l, j));
                componentwise_relerror =
                    max(componentwise_relerror, std::abs(C(i, j)) / aux);
            }
        }

        /* print the error */
        std::cout << "componentwise rel. error = " << componentwise_relerror
                  << std::endl;
        std::cout << "gammak = " << (k * u) / (1 - k * u) << std::endl;
    }

    /* free data */
    starpu_free_noflag(A_, (m + d1) * (k + d3) * sizeof(T));
    starpu_free_noflag(B_, (k + d3) * (n + d2) * sizeof(T));
    starpu_free_noflag(C_, (m + d1) * (n + d2) * sizeof(T));

    /* terminate StarPU */
    starpu_cublas_shutdown();
    starpu_shutdown();

    return 0;
}
