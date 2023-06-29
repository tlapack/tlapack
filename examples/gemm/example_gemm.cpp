/// @file example_gemm.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#ifdef USE_MPFR
    #include <tlapack/plugins/mpreal.hpp>
#endif

// <T>LAPACK
#include <tlapack/legacy_api/blas/gemm.hpp>
#include <tlapack/legacy_api/blas/nrm2.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <vector>

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using idx_t = size_t;
    using tlapack::min;
    using colmajor_matrix_t =
        tlapack::legacyMatrix<T, idx_t, tlapack::Layout::ColMajor>;
    using rowmajor_matrix_t =
        tlapack::legacyMatrix<T, idx_t, tlapack::Layout::RowMajor>;

    // Functors for creating new matrices
    tlapack::Create<colmajor_matrix_t> new_colmajor_matrix;
    tlapack::Create<rowmajor_matrix_t> new_rowmajor_matrix;

    // Column Major Matrices
    std::vector<T> A_;
    auto A = new_colmajor_matrix(A_, m, k);
    std::vector<T> B_;
    auto B = new_colmajor_matrix(B_, k, n);
    std::vector<T> C_;
    auto C = new_colmajor_matrix(C_, m, n);

    // Row Major Matrices
    std::vector<T> Ar_;
    auto Ar = new_rowmajor_matrix(Ar_, m, k);
    std::vector<T> Br_;
    auto Br = new_rowmajor_matrix(Br_, k, n);
    std::vector<T> Cr_;
    auto Cr = new_rowmajor_matrix(Cr_, m, n);

    // Number of runs to measure the minimum execution time
    int Nruns = 10;
    std::chrono::nanoseconds bestTime;

    // Initialize A and Ar with junk
    for (idx_t j = 0; j < k; ++j)
        for (idx_t i = 0; i < m; ++i) {
            A(i, j) = T(static_cast<float>(0xDEADBEEF));
            Ar(i, j) = A(i, j);
        }

    // Generate a random matrix in a submatrix of A
    for (idx_t j = 0; j < min(k, n); ++j)
        for (idx_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            Ar(i, j) = A(i, j);
        }

    // The diagonal of B is full of ones
    for (idx_t i = 0; i < min(k, n); ++i) {
        B(i, i) = 1.0;
        Br(i, i) = B(i, i);
    }

    // 1) Using legacy LAPACK API:

    bestTime = std::chrono::nanoseconds::max();
    for (int run = 0; run < Nruns; ++run) {
        // Set C using A
        for (idx_t j = 0; j < min(k, n); ++j)
            for (idx_t i = 0; i < m; ++i)
                C(i, j) = A(i, j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        // C = -1.0*A*B + 1.0*C
        tlapack::legacy::gemm(tlapack::Layout::ColMajor, tlapack::Op::NoTrans,
                              tlapack::Op::NoTrans, m, n, k, T(-1.0), &A_[0], m,
                              &B_[0], k, T(1.0), &C_[0], m);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Update best time
        if (elapsed < bestTime) bestTime = elapsed;
    }

    // Output
    std::cout << "Using legacy LAPACK API:" << std::endl
              << "||C-AB||_F = " << tlapack::legacy::nrm2(n, &C_[0], 1)
              << std::endl
              << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;

    // Using abstract interface:

    bestTime = std::chrono::nanoseconds::max();
    for (int run = 0; run < Nruns; ++run) {
        // Set C using A
        for (idx_t j = 0; j < min(k, n); ++j)
            for (idx_t i = 0; i < m; ++i)
                C(i, j) = A(i, j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        // C = -1.0*A*B + 1.0*C
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(-1.0), A, B,
                      T(1.0), C);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Update best time
        if (elapsed < bestTime) bestTime = elapsed;
    }

    // Output
    std::cout << "Using abstract interface:" << std::endl
              << "||C-AB||_F = " << tlapack::legacy::nrm2(n, &C_[0], 1)
              << std::endl
              << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;

    // Using abstract interface with row major layout:

    bestTime = std::chrono::nanoseconds::max();
    for (int run = 0; run < Nruns; ++run) {
        // Set C using A
        for (idx_t j = 0; j < min(k, n); ++j)
            for (idx_t i = 0; i < m; ++i)
                Cr(i, j) = Ar(i, j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        // C = -1.0*A*B + 1.0*C
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(-1.0), Ar,
                      Br, T(1.0), Cr);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Update best time
        if (elapsed < bestTime) bestTime = elapsed;
    }

    // Output
    std::cout << "Using abstract interface with row major layout:" << std::endl
              << "||C-AB||_F = " << tlapack::legacy::nrm2(n, &C_[0], 1)
              << std::endl
              << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, k;

    // Default arguments
    m = (argc < 2) ? 100 : atoi(argv[1]);
    n = (argc < 3) ? 200 : atoi(argv[2]);
    k = (argc < 4) ? 50 : atoi(argv[3]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float >( %d, %d, %d )\n", m, n, k);
    run<float>(m, n, k);
    printf("-----------------------\n");

    printf("run< double >( %d, %d, %d )\n", m, n, k);
    run<double>(m, n, k);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d, %d )\n", m, n, k);
    run<std::complex<float> >(m, n, k);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d, %d )\n", m, n, k);
    run<std::complex<double> >(m, n, k);
    printf("-----------------------\n");

#ifdef USE_MPFR
    printf("run< mpfr::mpreal >( %d, %d, %d )\n", m, n, k);
    run<mpfr::mpreal>(m, n, k);
    printf("-----------------------\n");

    printf("run< complex<mpfr::mpreal> >( %d, %d, %d )\n", m, n, k);
    run<std::complex<mpfr::mpreal> >(m, n, k);
    printf("-----------------------\n");
#endif

    return 0;
}
