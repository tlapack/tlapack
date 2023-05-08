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
#include <tlapack/lapack/getrf_recursive.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// C++ headers
#include <iostream>

int main(int argc, char** argv)
{
    using namespace tlapack;
    using starpu::Matrix;
    using starpu::idx_t;
    using T = float;

    srand(3);

    idx_t m = 14;
    idx_t n = 7;
    idx_t r = 14;
    idx_t s = 7;
    char method = '0';  // 'r' for recursive, '0' for level0

    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) r = atoi(argv[3]);
    if (argc > 4) s = atoi(argv[4]);
    if (argc > 5) method = tolower(argv[5][0]);
    if (argc > 6 || (method != 'r' && method != '0') || (m % r != 0) ||
        (n % s != 0) || (r > m) || (s > n) || (r == 0) || (s == 0) ||
        (r != m && method == 'r') || (s != n && method == 'r')) {
        std::cout << "Usage: " << argv[0] << " [m] [n] [r] [s] [method]"
                  << std::endl;
        std::cout << "  m:      number of rows of A (default: 14)" << std::endl;
        std::cout << "  n:      number of columns of A (default: 7)"
                  << std::endl;
        std::cout << "  r:      number of tiles in x (rows) direction "
                     "(default: 14)."
                  << "r=m if method is recursive."
                  << "Currently, m must be divisible by r." << std::endl;
        std::cout << "  s:      number of tiles in y (columns) direction "
                     "(default: 7)."
                  << "s=n if method is recursive."
                  << "Currently, n must be divisible by s." << std::endl;

        std::cout << "  method: 'r' for recursive, '0' for level0 (default: 0)"

                  << std::endl;
        return 1;
    }

    // Print input parameters
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "method = " << (method == 'r' ? "recursive" : "level0")
              << std::endl
              << std::endl;

    /* initialize StarPU */
    setenv("STARPU_CODELET_PROFILING", "0", 1);
    setenv("STARPU_SCHED", "dmdas", 1);
    setenv("HWLOC_COMPONENTS", "-gl", 1);
    const int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    {
        const idx_t k = std::min(m, n);

        /* create matrix A with m-by-n tiles */
        T* A_;
        starpu_malloc((void**)&A_, m * n * sizeof(T));
        for (idx_t i = 0; i < m * n; i++) {
            A_[i] = T((float)rand() / RAND_MAX);
        }
        Matrix<T> A(A_, m, n);
        A.create_grid(m, n);

        /* compute norm of A */
        const real_type<T> normA = lange(frob_norm, A);

        /* print matrix A */
        std::cout << "A = " << A << std::endl;
        assert(A.nrows() == m);
        assert(A.ncols() == n);

        /* Modify an entry of A*/
        if (m > 1 && n > 2) A(1, 2) = T(0);
        if (m > 3 && n > 3) A(3, 3) = A(0, 0);

        /* print modified matrix A */
        std::cout << "A = " << A << std::endl;

        /* create a copy of matrix A in Acopy with r-by-s tiles */
        T* Acopy_;
        starpu_malloc((void**)&Acopy_, m * n * sizeof(T));
        Matrix<T> Acopy(Acopy_, m, n);
        Acopy.create_grid(r, s);
        lacpy(dense, A, Acopy);

        /* print matrix Acopy */
        std::cout << "Acopy = " << Acopy << std::endl;

        /* create permutation vector with k tiles */
        idx_t* p_;
        starpu_malloc((void**)&p_, k * sizeof(idx_t));
        Matrix<idx_t> p(p_, k, 1);
        p.create_grid(k, 1);

        /* LU factorization */
        if (method == '0')
            getrf_level0(Acopy, p);
        else
            getrf_recursive(Acopy, p);
        std::cout << "LU = " << Acopy << std::endl;

        /* Create and print matrix L */
        T* L_;
        starpu_malloc((void**)&L_, m * k * sizeof(T));
        Matrix<T> L(L_, m, k);
        lacpy(lowerTriangle, Acopy, L);
        for (idx_t i = 0; i < k; ++i)
            L(i, i) = T(1);
        std::cout << "L = " << L << std::endl;

        /* Create and print matrix U */
        T* U_;
        starpu_malloc((void**)&U_, k * n * sizeof(T));
        Matrix<T> U(U_, k, n);
        lacpy(upperTriangle, Acopy, U);
        std::cout << "U = " << U << std::endl;

        /* print matrix L*U */
        if (m > n) {
            for (idx_t i = 0; i < m; ++i)
                for (idx_t j = i + 1; j < k; ++j)
                    L(i, j) = T(0);
            trmm(right_side, upperTriangle, noTranspose, nonUnit_diagonal, 1, U,
                 L);
            lacpy(dense, L, Acopy);
        }
        else {
            for (idx_t i = 0; i < k; ++i)
                for (idx_t j = 0; j < i; ++j)
                    U(i, j) = T(0);
            trmm(left_side, lowerTriangle, noTranspose, unit_diagonal, 1, L, U);
            lacpy(dense, U, Acopy);
        }
        std::cout << "L*U = " << Acopy << std::endl;

        /* Permute rows of A */
        for (idx_t i = 0; i < k; ++i) {
            if (p[i] != i) {
                auto Api = row(A, p[i]);
                auto Ai = row(A, i);
                tlapack::swap(Ai, Api);
            }
        }

        /* Verify the factorization is good */
        for (idx_t i = 0; i < A.nrows(); ++i) {
            for (idx_t j = 0; j < A.ncols(); j++) {
                A(i, j) -= Acopy(i, j);
            }
        }
        std::cout << "A - LU = " << A << std::endl;
        std::cout << "||A - LU||/||A|| = " << lange(frob_norm, A) / normA
                  << std::endl;

        starpu_free_noflag(L_, m * k * sizeof(T));
        starpu_free_noflag(U_, k * n * sizeof(T));
        starpu_free_noflag(p_, k * sizeof(idx_t));
        starpu_free_noflag(Acopy_, m * n * sizeof(T));
        starpu_free_noflag(A_, m * n * sizeof(T));
    }

    /* terminate StarPU */
    starpu_shutdown();

    return 0;
}
