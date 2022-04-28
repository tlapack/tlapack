/// @file test_schur_swap.cpp
/// @brief Test 1x1 and 2x2 sylvester solver
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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

TEST_CASE("1x1 swap gives correct result", "[utils]")
{

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t n1 = 1;
    const idx_t n2 = 1;
    const idx_t n = 5;
    const real_t eps = uroundoff<real_t>();

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _Q(new T[n * n]);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);

    std::unique_ptr<T[]> _A_copy(new T[n * n]);
    auto A_copy = colmajor_matrix<T>(&_A_copy[0], n, n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = zero;

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    schur_swap(true, A, Q, 1, 1, 1);

    T norm_orth_1, norm_repres_1;
    const bool verbose = false;
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

        CHECK(norm_orth_1 <= 1.0e2 * eps);

        if (verbose)
        {
            std::cout << std::endl
                      << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute Q*A_copyQ

    if (verbose)
    {
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto work = colmajor_matrix<T>(&_work[0], n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABC);

        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)0.0, A_copy);

        std::cout << std::endl
                  << "Q'A_copyQ = ";
        printMatrix(A_copy);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                A_copy(i, j) -= A(i, j);

        std::cout << std::endl
                  << "Q'A_copyQ - A = ";
        printMatrix(A_copy);

        // Compute ||Q'Q - I||_F
        norm_repres_1 = lange(frob_norm, A_copy);

        CHECK(norm_repres_1 <= 1.0e2 * eps);
    }
}

TEST_CASE("1x2 swap gives correct result", "[utils]")
{

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t j = 1;
    const idx_t n1 = 1;
    const idx_t n2 = 2;
    const idx_t n = 5;
    const real_t eps = uroundoff<real_t>();

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _Q(new T[n * n]);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);

    std::unique_ptr<T[]> _A_copy(new T[n * n]);
    auto A_copy = colmajor_matrix<T>(&_A_copy[0], n, n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = zero;

    if( n1 == 2)
        A( j + 1, j ) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (n2 == 2)
        A(j + n1 + 1, j + n1) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    schur_swap(true, A, Q, (idx_t)1, n1, n2);

    T norm_orth_1, norm_repres_1;
    const bool verbose = false;
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

        CHECK(norm_orth_1 <= 1.0e2 * eps);

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

        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)0.0, A_copy);

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

        CHECK(norm_repres_1 <= 1.0e2 * eps);
    }
}

TEST_CASE("2x1 swap gives correct result", "[utils]")
{

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t j = 1;
    const idx_t n1 = 2;
    const idx_t n2 = 1;
    const idx_t n = 5;
    const real_t eps = uroundoff<real_t>();

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _Q(new T[n * n]);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);

    std::unique_ptr<T[]> _A_copy(new T[n * n]);
    auto A_copy = colmajor_matrix<T>(&_A_copy[0], n, n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = zero;

    if( n1 == 2)
        A( j + 1, j ) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (n2 == 2)
        A(j + n1 + 1, j + n1) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    schur_swap(true, A, Q, (idx_t)1, n1, n2);

    T norm_orth_1, norm_repres_1;
    const bool verbose = true;
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

        CHECK(norm_orth_1 <= 1.0e2 * eps);

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

        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)0.0, A_copy);

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

        CHECK(norm_repres_1 <= 1.0e2 * eps);
    }
}

TEST_CASE("2x2 swap gives correct result", "[utils]")
{

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t j = 1;
    const idx_t n1 = 2;
    const idx_t n2 = 2;
    const idx_t n = 6;
    const real_t eps = uroundoff<real_t>();

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _Q(new T[n * n]);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);

    std::unique_ptr<T[]> _A_copy(new T[n * n]);
    auto A_copy = colmajor_matrix<T>(&_A_copy[0], n, n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = zero;

    if( n1 == 2)
        A( j + 1, j ) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (n2 == 2)
        A(j + n1 + 1, j + n1) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    schur_swap(true, A, Q, (idx_t)1, n1, n2);

    T norm_orth_1, norm_repres_1;
    const bool verbose = false;
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

        CHECK(norm_orth_1 <= 1.0e2 * eps);

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

        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)0.0, A_copy);

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

        CHECK(norm_repres_1 <= 1.0e2 * eps);
    }
}

TEST_CASE("complex swap gives correct result", "[utils]")
{

    typedef std::complex<float> T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t j = 1;
    const idx_t n1 = 1;
    const idx_t n2 = 1;
    const idx_t n = 6;
    const real_t eps = uroundoff<real_t>();

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _Q(new T[n * n]);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);

    std::unique_ptr<T[]> _A_copy(new T[n * n]);
    auto A_copy = colmajor_matrix<T>(&_A_copy[0], n, n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = zero;

    if( n1 == 2)
        A( j + 1, j ) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (n2 == 2)
        A(j + n1 + 1, j + n1) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    schur_swap(true, A, Q, (idx_t)1, n1, n2);

    real_t norm_orth_1, norm_repres_1;
    const bool verbose = false;
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

        CHECK(norm_orth_1 <= 1.0e2 * eps);

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

        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)0.0, A_copy);

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

        CHECK(norm_repres_1 <= 1.0e2 * eps);
    }
}