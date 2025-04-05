/// @file examples/starpu/example_potrf.cpp
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
#include <tlapack/lapack/lahqr.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/starpu/starpu.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>

using tlapack::starpu::idx_t;

template <class T>
int run(idx_t n, idx_t nx, bool check_error = false)
{
    using namespace tlapack;
    using starpu::Matrix;
    using tlapack::starpu::idx_t;
    using real_t = real_type<T>;
    using complex_t = complex_type<real_t>;

    // constant parameters
    const real_t one(1);
    const real_t zero(0);

    /* create arrays A, Q, H and s */
    T *A_, *H_, *Q_;
    complex_t* s_;
    starpu_malloc((void**)&H_, n * n * sizeof(T));
    starpu_malloc((void**)&Q_, n * n * sizeof(T));
    starpu_malloc((void**)&s_, n * sizeof(complex_t));
    if (check_error) starpu_malloc((void**)&A_, n * n * sizeof(T));

    // H is upper Hessenberg
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < min(n, j + 2); i++)
            if constexpr (is_complex<T>)
                H_[i + j * n] = T((float)rand() / (float)RAND_MAX,
                                  (float)rand() / (float)RAND_MAX);
            else
                H_[i + j * n] = T((float)rand() / (float)RAND_MAX);
        for (idx_t i = j + 2; i < n; ++i)
            H_[i + j * n] = zero;
    }

    // Q is identity
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++)
            Q_[i + j * n] = zero;
        Q_[j + j * n] = one;
    }

    if (check_error) {
        for (idx_t j = 0; j < n; j++)
            for (idx_t i = 0; i < n; i++)
                A_[i + j * n] = H_[i + j * n];
    }

    // Errors
    real_t normQHQ = 0;
    real_t normQQH = 0;
    real_t errorSimilarityTransform = 0;

    std::chrono::nanoseconds elapsed_time;
    {
        /* create matrices H, Q and vector s */
        Matrix<T> H(H_, n, n, nx, nx);
        std::cout << "H = " << H << std::endl;
        Matrix<T> Q(Q_, n, n, nx, nx);
        std::cout << "Q = " << Q << std::endl;
        Matrix<complex_t> s(s_, n, 1, nx, 1);
        std::cout << "s = " << s << std::endl;

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        /* call multishift QR */
        int info = lahqr(true, true, 0, n, H, s, Q);

        // Record end time
        starpu_task_wait_for_all();
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        elapsed_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        if (info != 0)
            std::cout << "LAHQR ended with info " << info << std::endl;

        if (check_error) {
            T* E_;
            starpu_malloc((void**)&E_, n * n * sizeof(T));
            Matrix<T> E(E_, n, n, nx, nx);

            // E = Q^H Q - I
            herk(Uplo::Upper, Op::ConjTrans, (real_t)1.0, Q, E);
            for (idx_t j = 0; j < n; ++j)
                E(j, j) -= one;

            // ||Q^H Q - I||_F
            normQHQ = tlapack::lange(tlapack::FROB_NORM, E);

            // E = Q Q^H - I
            herk(Uplo::Upper, Op::NoTrans, (real_t)1.0, Q, E);
            for (idx_t j = 0; j < n; ++j)
                E(j, j) -= one;

            // ||Q Q^H - I||_F
            normQQH = tlapack::lange(tlapack::FROB_NORM, E);

            // Clean the lower triangular part that was used a workspace
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j + 2; i < n; ++i)
                    H(i, j) = zero;

            Matrix<T> A(A_, n, n, nx, nx);

            // E = Q^H H Q - T
            gemm(Op::ConjTrans, Op::NoTrans, (real_t)1.0, Q, A, E);
            gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, E, Q, (real_t)-1.0, H);

            // ||Q^H H Q - T||_F / ||H||_F
            errorSimilarityTransform = tlapack::lange(tlapack::FROB_NORM, H) /
                                       tlapack::lange(tlapack::FROB_NORM, A);
        }

        std::cout << "H = " << H << std::endl;
    }

    // Output
    std::cout << "Q T Q^H = H   =>   "
              << "||Q^H Q - I||_F = " << ((check_error) ? normQHQ : real_t(-1))
              << std::endl
              << "||Q Q^H - I||_F = " << ((check_error) ? normQQH : real_t(-1))
              << std::endl
              << "||Q^H H Q - T||_F = "
              << ((check_error) ? errorSimilarityTransform : real_t(-1))
              << std::endl
              << "time = " << elapsed_time.count() * 1.0e-9 << " s"
              << std::endl;

    // Clean up
    starpu_free_noflag(H_, n * n * sizeof(T));
    if (check_error) starpu_free_noflag(A_, n * n * sizeof(T));

    return 0;
}

int main(int argc, char** argv)
{
    // initialize random seed
    srand(3);

    idx_t n = 10;
    idx_t nx = 10;
    int precision = 0b1111;
    bool check_error = false;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) nx = atoi(argv[2]);
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
    if (argc > 5 || (n % nx != 0) || (nx > n) || (n <= 0) || (nx <= 0) ||
        precision == 0) {
        std::cout << "Usage: " << argv[0]
                  << " [n] [nx] [precision] [check_error]" << std::endl;
        std::cout << "  n:      number of rows and columns of A (default: 50)"
                  << std::endl;
        std::cout << "  nx:      number of tiles in x and y directions of A "
                     "(default: 10)."
                  << std::endl;
        std::cout << "  precision: s (0b0001), d (0b0010), c (0b0100), z "
                     "(0b1000), a (0b1111) (default: all (0b1111))."
                  << std::endl;
        std::cout << "  check_error: yes or no (default: no)" << std::endl;
        return -1;
    }

    // Print input parameters
    std::cout << "n = " << n << std::endl;
    std::cout << "nx = " << nx << std::endl << std::endl;

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
        if (run<float>(n, nx, check_error)) return 1;
    }

    if (precision & 0b0010) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "double:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<double>(n, nx, check_error)) return 2;
    }

    if (precision & 0b0100) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "complex<float>:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<std::complex<float>>(n, nx, check_error)) return 3;
    }

    if (precision & 0b1000) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "complex<double>:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<std::complex<double>>(n, nx, check_error)) return 4;
    }

    /* terminate StarPU */
    starpu_cusolver_shutdown();
    starpu_cublas_shutdown();
    starpu_shutdown();

    return 0;
}
