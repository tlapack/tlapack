/// @file test_utils.cpp
/// @brief Test utils from <T>LAPACK.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <plugins/tlapack_stdvector.hpp>

#include <catch2/catch.hpp>
#include <tlapack.hpp>

#include <iostream>
#include <iomanip>

using namespace blas;
using namespace lapack;

// This should really be moved to test utils or something
template <typename matrix_t>
inline void printMatrix(const matrix_t &A)
{
    using idx_t = blas::size_type<matrix_t>;
    const idx_t m = blas::nrows(A);
    const idx_t n = blas::ncols(A);

    for (idx_t i = 0; i < m; ++i)
    {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(16) << A(i, j) << " ";
    }
}

TEST_CASE("Multishift sweep", "[eigenvalues]")
{

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t n = 15;
    const idx_t n_shifts = 4;
    const real_t eps = uroundoff<real_t>();

    const bool verbose = true;

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _A_copy(new T[n * n]);
    auto A_copy = colmajor_matrix<T>(&_A_copy[0], n, n);

    std::unique_ptr<T[]> _Q(new T[n * n]);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);

    std::unique_ptr<T[]> _V(new T[3 * (n_shifts/2)]);
    auto V = colmajor_matrix<T>(&_V[0], 3, (n_shifts/2));

    auto s = std::vector<complex_t>(n_shifts);
    for (int i = 0; i < n_shifts; ++i)
    {
        s[i] = complex_t(i + 1, 0.);
    }

    // Generate a random upper Hessenberg matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < std::min(n, j + 2); ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            A(i, j) = zero;

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    // Print original A
    if (verbose)
    {
        printMatrix(A);
    }

    auto normA = lange(lapack::frob_norm, A);

    multishift_QR_sweep(true, true, 0, n, A, s, Q, V);

    // Clean the lower triangular part that was used a workspace
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            A(i, j) = zero;

    // Print Q and A
    if (verbose)
    {
        std::cout << std::endl
                  << "Q = ";
        printMatrix(Q);
        std::cout << std::endl
                  << "A = ";
        printMatrix(A);
    }

    real_t norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto work = colmajor_matrix<T>(&_work[0], n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        laset(Uplo::General, (T)0.0, (T)1.0, work);
        // work receives Q'Q - I
        // blas::syrk( blas::Uplo::Upper, blas::Op::ConjTrans, (T) 1.0, Q, (T) -1.0, work );
        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, Q, (T)-1.0, work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 = lansy(frob_norm, Uplo::Upper, work);

        CHECK(norm_orth_1 <= 1.0e2 * n * eps);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute Q*A_copyQ

    {
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto work = colmajor_matrix<T>(&_work[0], n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABC);

        blas::gemm(blas::Op::ConjTrans, blas::Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, (T)1.0, work, Q, (T)0.0, A_copy);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'A_copyQ = ";
            printMatrix(A_copy);
        }

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                A_copy(i, j) -= A(i, j);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'A_copyQ - A = ";
            printMatrix(A_copy);
        }

        // Compute ||Q'Q - I||_F
        norm_repres_1 = lange(frob_norm, A_copy);

        CHECK(norm_repres_1 <= 1.0e2 * n * eps);
    }

    if (verbose)
        std::cout << std::endl;
}

TEST_CASE("AED", "[eigenvalues]")
{

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t n = 12;
    const idx_t window_size = 4;
    const real_t eps = uroundoff<real_t>();

    const bool verbose = true;

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _A_copy(new T[n * n]);
    auto A_copy = colmajor_matrix<T>(&_A_copy[0], n, n);

    std::unique_ptr<T[]> _Q(new T[n * n]);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);

    std::unique_ptr<complex_t[]> _s(new complex_t[n]);
    auto s = legacyVector<complex_t>(n, &_s[0]);

    // Generate a random upper Hessenberg matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < std::min(n, j + 2); ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            A(i, j) = zero;

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    // Print original A
    if (verbose)
    {
        printMatrix(A);
    }

    auto normA = lange(lapack::frob_norm, A);

    idx_t ns, nd;

    agressive_early_deflation(true, true, (idx_t)0, n, window_size, A, s, Q, ns, nd);

    // Clean the lower triangular part that was used a workspace
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            A(i, j) = zero;

    // Print Q and A
    if (verbose)
    {
        std::cout << std::endl
                  << "Q = ";
        printMatrix(Q);
        std::cout << std::endl
                  << "A = ";
        printMatrix(A);
    }

    real_t norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto work = colmajor_matrix<T>(&_work[0], n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        laset(Uplo::General, (T)0.0, (T)1.0, work);
        // work receives Q'Q - I
        // blas::syrk( blas::Uplo::Upper, blas::Op::ConjTrans, (T) 1.0, Q, (T) -1.0, work );
        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, Q, (T)-1.0, work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 = lansy(frob_norm, Uplo::Upper, work);

        CHECK(norm_orth_1 <= 1.0e2 * n * eps);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute Q*A_copyQ

    {
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto work = colmajor_matrix<T>(&_work[0], n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABC);

        blas::gemm(blas::Op::ConjTrans, blas::Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, (T)1.0, work, Q, (T)0.0, A_copy);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'A_copyQ = ";
            printMatrix(A_copy);
        }

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                A_copy(i, j) -= A(i, j);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'A_copyQ - A = ";
            printMatrix(A_copy);
        }

        // Compute ||Q'Q - I||_F
        norm_repres_1 = lange(frob_norm, A_copy);

        CHECK(norm_repres_1 <= 1.0e2 * n * eps);
    }

    if (verbose)
        std::cout << std::endl;
}