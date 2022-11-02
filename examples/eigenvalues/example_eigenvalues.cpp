/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack.hpp>

#include <memory>
#include <vector>
#include <chrono> // for high_resolution_clock
#include <iostream>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
inline void printMatrix(const matrix_t &A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i)
    {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::legacyMatrix<T>;
    using std::size_t;

    // Functor for creating new matrices of type matrix_t
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<T> tau(n);

    // Matrix views
    std::vector<T> A_; auto A = new_matrix(A_, n, n);
    std::vector<T> H_; auto H = new_matrix(H_, n, n);
    std::vector<T> Q_; auto Q = new_matrix(Q_, n, n);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j)
    {
        for (size_t i = 0; i < n; ++i)
        {
            A(i, j) = static_cast<float>(0xDEADBEEF);
            Q(i, j) = static_cast<float>(0xCAFED00D);
        }
        for (size_t i = 0; i < n; ++i)
        {
            H(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::frob_norm, A);

    // Print A
    if (verbose)
    {
        std::cout << std::endl
                  << "A = ";
        printMatrix(A);
    }

    // Copy A to Q
    tlapack::lacpy(tlapack::Uplo::General, A, Q);

    // 1) Compute A = QHQ* (Stored in the matrix Q)

    // Record start time
    auto startQHQ = std::chrono::high_resolution_clock::now();
    {
        // Hessenberg factorization
        int err = tlapack::gehrd(0, n, Q, tau);
        tlapack_check_false(err);
    }
    // Record end time
    auto endQHQ = std::chrono::high_resolution_clock::now();


    // Save the H matrix
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < std::min(n, j + 2); ++i)
            H(i, j) = Q(i, j);

    // Record start time
    auto startQ = std::chrono::high_resolution_clock::now();
    {
        // Generate Q = H_1 H_2 ... H_n
        int err = tlapack::unghr(0, n, Q, tau);
        tlapack_check_false(err);
    }
    // Record end time
    auto endQ = std::chrono::high_resolution_clock::now();


    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Record start time
    auto startSchur = std::chrono::high_resolution_clock::now();
    {
        // Shur factorization
        std::vector<std::complex<real_t>> w(n);
        int err = tlapack::multishift_qr(true, true, 0, n, H, w, Q);
        tlapack_check_false(err);
    }
    // Record end time
    auto endSchur = std::chrono::high_resolution_clock::now();

    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Compute elapsed time in nanoseconds
    auto elapsedQHQ = std::chrono::duration_cast<std::chrono::nanoseconds>(endQHQ - startQHQ);
    auto elapsedQ = std::chrono::duration_cast<std::chrono::nanoseconds>(endQ - startQ);
    auto elapsedSchur = std::chrono::duration_cast<std::chrono::nanoseconds>(endSchur - startSchur);

    // Print Q and H
    if (verbose)
    {
        std::cout << std::endl
                  << "Q = ";
        printMatrix(Q);
        std::cout << std::endl
                  << "H = ";
        printMatrix(H);
    }

    real_t norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::Uplo::General, (T)0.0, (T)1.0, work);
        // work receives Q'Q - I
        // tlapack::syrk( tlapack::Uplo::Upper, tlapack::Op::ConjTrans, (T) 1.0, Q, (T) -1.0, work );
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, (T)1.0, Q, Q, (T)-1.0, work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 = tlapack::lansy(tlapack::frob_norm, tlapack::Uplo::Upper, work);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute ||QHQ* - A||_F / ||A||_F

    std::vector<T> Hcopy_;
    auto H_copy = new_matrix(Hcopy_, n, n);
    tlapack::lacpy(tlapack::Uplo::General,H, H_copy);
    {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABC);

        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, Q, H, work);
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans, (T)1.0, work, Q, H);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                H(i, j) -= A(i, j);

        if (verbose)
        {
            std::cout << std::endl
                      << "QHQ'-A = ";
            printMatrix(H);
        }

        norm_repres_1 = tlapack::lange(tlapack::frob_norm, H) / normA;
    }

    // 4) Compute Q*AQ (usefull for debugging)

    if(verbose){
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABC);

        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, (T)1.0, Q, A, work);
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, work, Q, A);

        std::cout << std::endl
                    << "Q'AQ = ";
        printMatrix(A);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                A(i, j) -= H_copy(i, j);

        std::cout << std::endl
                    << "Q'AQ - H = ";
        printMatrix(A);
    }

    std::cout << std::endl;
    std::cout << "Hessenberg time = " << elapsedQHQ.count() * 1.0e-9 << " s" << std::endl;
    std::cout << "Q forming time = "<< elapsedQ.count() * 1.0e-9 << " s" << std::endl;
    std::cout << "QR time = " << elapsedSchur.count() * 1.0e-9 << " s" << std::endl;
    std::cout << "||QHQ* - A||_F/||A||_F  = " << norm_repres_1
              << ",        ||Q'Q - I||_F  = " << norm_orth_1;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int n;

    // Default arguments
    n = (argc < 2) ? 100 : atoi(argv[1]);

    srand(3); // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float >( %d )", n);
    run<float>(n);
    printf("-----------------------\n");

    printf("run< std::complex<float>  >( %d )", n);
    run<std::complex<float>>(n);
    printf("-----------------------\n");

    printf("run< double >( %d )", n);
    run<double>(n);
    printf("-----------------------\n");

    printf("run< std::complex<double>  >( %d )", n);
    run<std::complex<double>>(n);
    printf("-----------------------\n");

    printf("run< long double >( %d )", n);
    run<long double>(n);
    printf("-----------------------\n");

    printf("run< std::complex<long double>  >( %d )", n);
    run<std::complex<long double>>(n);
    printf("-----------------------\n");

    return 0;
}
