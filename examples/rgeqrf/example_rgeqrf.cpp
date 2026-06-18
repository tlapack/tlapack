/// @file example_rgeqrf.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// rgeqrf utilizes geqrt3 to complete a QR factorization with a repeatedly
// halving block size as it moves to the right
//
// rgeqrf does not compute the full T matrix

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/rgeqrf.hpp>
#include <tlapack/lapack/ung2r.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<T> tau(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<T> Tmatrix_;
    auto Tmatrix = new_matrix(Tmatrix_, n, n);

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A(i, j) = T(static_cast<float>(0xDEADBEEF));
            Q(i, j) = T(static_cast<float>(0xCAFED00D));
        }
        for (idx_t i = 0; i < n; ++i) {
            Tmatrix(i, j) = T(static_cast<float>(0XFEE1DEAD));
            R(i, j) = T(static_cast<float>(0xFEE1DEAD));
        }
        tau[j] = T(static_cast<float>(0xFFBADD11));
    }

    // Generate a random matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));

    // Frobenius norm of A
    real_t normA = tlapack::lange(tlapack::FROB_NORM, A);

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A);
        std::cout << std::endl;
    }

    // Copy A to Q
    tlapack::lacpy(tlapack::GENERAL, A, Q);

    // 1) Compute A = QR (Stored in the matrix Q)

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // QR Factorization
        tlapack::rgeqrf(Q, Tmatrix);
    }
    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Compute FLOPS **WIP** Needs to be calculated
    double flops = (3.0 * m * n * n) - (5.0 / 6.0 * n * n * n) +
                   (1.0 / 2.0 * n * n) + (1.0 / 3.0 * n);

    double flopsQR = flops / (elapsedQR.count() * 1.0e-9);

    // Save the R matrix
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);

    // Fill tau with the diagonal of the T matrix
    for (idx_t i = 0; i < n; ++i) {
        tau[i] = Tmatrix(i, i);
    }

    // Generates Q = H_1 H_2... H_n
    tlapack::ung2r(Q, tau);

    // Print Q and R
    if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix(Q);
        std::cout << std::endl << "R = ";
        printMatrix(R);
        std::cout << std::endl << "T = ";
        printMatrix(Tmatrix);
        std::cout << std::endl;
    }

    // 2) Compute ||Q'Q - I||_F

    real_t norm_orth, norm_repres;

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::UPPER_TRIANGLE, static_cast<T>(0.0),
                       static_cast<T>(1.0), work);
        // work receives Q'Q - I
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                      static_cast<T>(1.0), Q, Q, static_cast<T>(-1.0), work);

        // Compute ||Q'Q - I||_F
        norm_orth =
            tlapack::lansy(tlapack::FROB_NORM, tlapack::UPPER_TRIANGLE, work);

        if (verbose) {
            std::cout << std::endl << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute ||QR - A||_F / ||A||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, m, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // Copy Q to work
        tlapack::lacpy(tlapack::GENERAL, Q, work);

        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit,
                      static_cast<T>(1.0), R, work);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                work(i, j) -= A(i, j);

        norm_repres = tlapack::lange(tlapack::FROB_NORM, work) / normA;
    }

    // *) Output

    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
              << ",   GFlop/sec = " << flopsQR * 1.0e-9
              << ", Flops = " << flops;
    std::cout << std::endl;
    std::cout << "NOTE: FLOPS ARE CURRENTLY WRONG" << std::endl;

    std::cout << "||QR - A||_F/||A||_F  = " << std::real(norm_repres)
              << ",        ||Q'Q - I||_F  = " << std::real(norm_orth);
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n;

    // Default arguments
    m = (argc < 2) ? 7 : atoi(argv[1]);
    n = (argc < 3) ? 5 : atoi(argv[2]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", m, n);
    run<long double>(m, n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )", m, n);
    run<std::complex<float>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )", m, n);
    run<std::complex<double>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<long double> >( %d, %d )", m, n);
    run<std::complex<long double>>(m, n);
    printf("-----------------------\n");

    return 0;
}
