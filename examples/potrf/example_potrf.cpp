/// @file example_gemm.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/potrf2.hpp>
#include <tlapack/lapack/potrs.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <vector>

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

inline lapack_int LAPACKE_xpotrf2(
    int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda)
{
    return LAPACKE_spotrf2(matrix_layout, uplo, n, a, lda);
}
inline lapack_int LAPACKE_xpotrf2(
    int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda)
{
    return LAPACKE_dpotrf2(matrix_layout, uplo, n, a, lda);
}
inline lapack_int LAPACKE_xpotrf2(int matrix_layout,
                                  char uplo,
                                  lapack_int n,
                                  std::complex<float>* a,
                                  lapack_int lda)
{
    return LAPACKE_cpotrf2(matrix_layout, uplo, n,
                           reinterpret_cast<complex*>(a), lda);
}
inline lapack_int LAPACKE_xpotrf2(int matrix_layout,
                                  char uplo,
                                  lapack_int n,
                                  std::complex<double>* a,
                                  lapack_int lda)
{
    return LAPACKE_zpotrf2(matrix_layout, uplo, n,
                           reinterpret_cast<dComplex*>(a), lda);
}

using idx_t = lapack_int;

//------------------------------------------------------------------------------
template <typename T>
void run(idx_t n)
{
    using namespace tlapack;
    using real_t = real_type<T>;

    // Matrix A
    std::vector<T> A_(n * n);
    legacyMatrix<T> A(n, n, &A_[0], n);

    // // Flops
    // const real_t nFlops = real_t(n*n*n) / 3;

    // Fill A with random entries
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Turn the upper part of A into a symmetric positive definite matrix
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < j; ++i)
            A(i, j) = T(0.5) * (A(i, j) + A(j, i));
        A(j, j) += n;
    }

    // 1) Using <T>LAPACK API:
    {
        std::vector<T> U_(n * n);
        legacyMatrix<T> U(n, n, &U_[0], n);

        // Put garbage on U_
        for (idx_t j = 0; j < n * n; ++j)
            if constexpr (is_complex<T>)
                U_[j] = T(static_cast<float>(0xDEADBEEF),
                          static_cast<float>(0xDEADBEEF));
            else
                U_[j] = T(static_cast<float>(0xDEADBEEF));

        // U_ receives the upper part of A
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i <= j; ++i)
                U_[i + j * n] = A(i, j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        int info = potrf2(upperTriangle, U);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        if (info != 0) {
            std::cout << "Cholesky ended with info " << info << std::endl;
        }

        // Solve U^H U R = A
        std::vector<T> R_(n * n);
        legacyMatrix<T> R(n, n, &R_[0], n);
        lacpy(dense, A, R);
        potrs(upperTriangle, U, R);

        // error = ||R-Id||_F / ||Id||_F
        for (idx_t i = 0; i < n; ++i)
            R(i, i) -= T(1);
        real_t error = lange(frob_norm, R) / std::sqrt(n);

        // Output
        std::cout << "Using <T>LAPACK:" << std::endl
                  << "U^H U R = A   =>   ||R-Id||_F / ||Id||_F = " << error
                  << std::endl
                  << "time = " << elapsed.count() * 1.0e-9 << " s" << std::endl;
        // std::cout << nFlops / elapsed.count() << " ";
    }

    // 2) Using LAPACKE:
    {
        std::vector<T> U(n * n);

        // Put garbage on U
        for (idx_t j = 0; j < n * n; ++j)
            if constexpr (is_complex<T>)
                U[j] = T(static_cast<float>(0xDEADBEEF),
                         static_cast<float>(0xDEADBEEF));
            else
                U[j] = T(static_cast<float>(0xDEADBEEF));

        // U receives the upper part of A
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i <= j; ++i)
                U[i + j * n] = A(i, j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        lapack_int info =
            LAPACKE_xpotrf2(LAPACK_COL_MAJOR, 'U', n, U.data(), n);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        if (info != 0) {
            std::cout << "Cholesky ended with info " << info << std::endl;
        }

        // Solve U^H U R = A
        std::vector<T> R_(n * n);
        legacyMatrix<T> R(n, n, &R_[0], n);
        lacpy(dense, A, R);
        potrs(upperTriangle, legacyMatrix<T>(n, n, &U[0], n), R);

        // error = ||R-Id||_F / ||Id||_F
        for (idx_t i = 0; i < n; ++i)
            R(i, i) -= T(1);
        real_t error = lange(frob_norm, R) / std::sqrt(n);

        // Output
        std::cout << "Using LAPACKE:" << std::endl
                  << "U^H U R = A   =>   ||R-Id||_F / ||Id||_F = " << error
                  << std::endl
                  << "time = " << elapsed.count() * 1.0e-9 << " s" << std::endl;
        // std::cout << nFlops / elapsed.count() << " ";
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    idx_t n;

    // Default arguments
    n = (argc < 2) ? 100 : atoi(argv[1]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    std::cout << "run< float >( " << n << " )" << std::endl;
    run<float>(n);
    std::cout << "-----------------------" << std::endl;

    std::cout << "run< double >( " << n << " )" << std::endl;
    run<double>(n);
    std::cout << "-----------------------" << std::endl;

    std::cout << "run< complex<float> >( " << n << " )" << std::endl;
    run<std::complex<float> >(n);
    std::cout << "-----------------------" << std::endl;

    std::cout << "run< complex<double> >( " << n << " )" << std::endl;
    run<std::complex<double> >(n);
    std::cout << "-----------------------" << std::endl;

    // std::cout << std::endl;

    return 0;
}
