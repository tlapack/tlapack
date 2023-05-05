/// @file examples/starpu/compare_with_mkl.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <complex>
#include <iostream>

// LAPACKE headers
#ifndef USE_MKL
// LAPACKE
extern "C" {
    #include <lapacke.h>
}
typedef float _Complex complex;
typedef double _Complex dComplex;
#else
// MKL LAPACKE
extern "C" {
    #include <mkl_lapacke.h>
}
typedef _MKL_Complex8 complex;
typedef _MKL_Complex16 dComplex;
#endif

using idx_t = lapack_int;

inline idx_t LAPACKE_xpotrf(
    int matrix_layout, char uplo, idx_t n, float* a, idx_t lda)
{
    return LAPACKE_spotrf(matrix_layout, uplo, n, a, lda);
}
inline idx_t LAPACKE_xpotrf(
    int matrix_layout, char uplo, idx_t n, double* a, idx_t lda)
{
    return LAPACKE_dpotrf(matrix_layout, uplo, n, a, lda);
}
inline idx_t LAPACKE_xpotrf(
    int matrix_layout, char uplo, idx_t n, std::complex<float>* a, idx_t lda)
{
    return LAPACKE_cpotrf(matrix_layout, uplo, n, reinterpret_cast<complex*>(a),
                          lda);
}
inline idx_t LAPACKE_xpotrf(
    int matrix_layout, char uplo, idx_t n, std::complex<double>* a, idx_t lda)
{
    return LAPACKE_zpotrf(matrix_layout, uplo, n,
                          reinterpret_cast<dComplex*>(a), lda);
}

template <class T>
int run(idx_t n)
{
    /* create arrays A and B */
    T* A_ = (T*)malloc(n * n * sizeof(T));

    /* A is symmetric positive definite and B is a copy of A */
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = 0; i < j; i++)
            A_[i + j * n] = A_[i * n + j] = T((float)rand() / (float)RAND_MAX);
    for (idx_t i = 0; i < n; i++)
        A_[i + n * i] += n;

    std::chrono::nanoseconds elapsed_time;
    {
        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        /* call potrf */
        idx_t info = LAPACKE_xpotrf(LAPACK_COL_MAJOR, 'U', n, A_, n);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        elapsed_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        if (info != 0)
            std::cout << "Cholesky ended with info " << info << std::endl;
    }

    // Output
    std::cout << "time = " << elapsed_time.count() * 1.0e-9 << " s"
              << std::endl;

    // Clean up
    free(A_);

    return 0;
}

int main(int argc, char** argv)
{
    // initialize random seed
    srand(3);

    idx_t n = 100;
    int precision = 0b1111;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) {
        char p = tolower(argv[2][0]);
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

    if (argc > 3 || (n <= 0) || precision == 0) {
        std::cout << "Usage: " << argv[0] << " [n] [precision]" << std::endl;
        std::cout << "  n:      number of rows and columns of A (default: 50)"
                  << std::endl;
        std::cout << "  precision: s (0b0001), d (0b0010), c (0b0100), z "
                     "(0b1000), a (0b1111) (default: all (0b1111))."
                  << std::endl;
        return -1;
    }

    // Print input parameters
    std::cout << "n = " << n << std::endl;

    // Run tests:

    if (precision & 0b0001) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "float:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<float>(n)) return 1;
    }

    if (precision & 0b0010) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "double:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<double>(n)) return 2;
    }

    if (precision & 0b0100) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "complex<float>:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<std::complex<float>>(n)) return 3;
    }

    if (precision & 0b1000) {
        std::cout << std::endl
                  << "-----------------------------------------------";
        std::cout << std::endl << "complex<double>:";
        std::cout << std::endl
                  << "-----------------------------------------------"
                  << std::endl;
        if (run<std::complex<double>>(n)) return 4;
    }

    return 0;
}
