/// @file example_geqrt3.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// DGEQRT3 recursively computes a QR factorization of a real M-by-N
// matrix A, using the compact WY representation of Q.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqrt3.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/larfb.hpp>
#include <tlapack/lapack/laset.hpp>

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
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> V_;
    auto V = new_matrix(V_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<T> T_;
    auto Tmatrix = new_matrix(T_, n, n);

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

    // Copy A to Q
    tlapack::lacpy(tlapack::GENERAL, A, Q);

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        tlapack::geqrt3(Q, Tmatrix);
    }
    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Compute the Frobenius norm of A
    auto normA = tlapack::lange(tlapack::FROB_NORM, A);

    T norm_orth, norm_repres;

    // 2) Compute ||Qᴴ Q - I||ꜰ

    {
        // Copy Upper Triangle of A into R
        tlapack::lacpy(tlapack::Uplo::Upper, Q, R);

        // Copy the Householder vectors into V
        tlapack::lacpy(tlapack::GENERAL, Q, V);

        // Q becomes the identity matrix
        laset(tlapack::GENERAL, static_cast<T>(0.0), static_cast<T>(1.0), Q);

        tlapack::larfb(tlapack::Side::Left, tlapack::Op::NoTrans,
                       tlapack::Direction::Forward, tlapack::StoreV::Columnwise,
                       V, Tmatrix, Q);

        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);
        ;

        // work receives the identity n*n
        tlapack::laset(tlapack::GENERAL, static_cast<T>(0.0),
                       static_cast<T>(1.0), work);
        // work receives Qᴴ Q - I
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                      static_cast<T>(1.0), Q, Q, static_cast<T>(-1.0), work);

        // Compute ||Qᴴ Q - I||ꜰ
        norm_orth = tlapack::lange(tlapack::FROB_NORM, work);

        if (verbose) {
            std::cout << std::endl << "QᴴQ-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute ||QR - A||ꜰ / ||A||ꜰ
    {
        std::vector<T> work_;

        auto work = new_matrix(work_, m, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                work(i, j) = static_cast<float>(0);
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
    double seconds = elapsedQR.count() * 1.0e-9;

    //(3*m*n² - 5/6*n³)
    double geqrt3_flops =
        (3.0 * ((double)m) * ((double)n) * ((double)n)) -
        ((5.0 / 6.0) * ((double)n) * ((double)n) * ((double)n));

    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms" << std::endl
              << "geqrt3 Flop/sec = " << (geqrt3_flops / seconds) * 1.0e-9
              << std::endl;

    //( 2 * m * n² - 2 / 3 * n³ ))
    auto geqr2_flops = ((2.0 * ((double)m) * ((double)n) * ((double)n)) -
                        (2.0 / 3.0 * ((double)n) * ((double)n) * ((double)n)));
    std::cout << "GEQR2 flops/sec = " << (geqr2_flops / seconds) * 1.0e-9
              << std::endl;

    // ( 3 * m * n² - 5 / 6 * n³ ) / ( 2 * m * n² - 2 / 3 * n³ ))
    auto geqr_ratio = ((3.0 * ((double)m) * ((double)n) * ((double)n)) -
                       (5.0 / 6.0 * ((double)n) * ((double)n) * ((double)n))) /
                      ((2.0 * ((double)m) * ((double)n) * ((double)n)) -
                       (2.0 / 3.0 * ((double)n) * ((double)n) * ((double)n)));

    std::cout << "GEQRT3/GEQR2 flop ratio = " << (geqr_ratio) << std::endl;
    std::cout << "||QR - A||ꜰ/||A||ꜰ  = " << std::real(norm_repres)
              << ",        ||QᴴQ - I||ꜰ  = " << std::real(norm_orth);
    std::cout << std::endl;
}
//================================================================================
//================================================================================
int main(int argc, char** argv)
{
    int m, n;

    // Default arguments
    m = (argc < 2) ? 1 : atoi(argv[1]);
    n = (argc < 3) ? 1 : atoi(argv[2]);

    srand(3);  // Init random seed

    m = 73;
    n = 55;

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
