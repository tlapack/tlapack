/// @file test_unmhr.cpp
/// @brief Test Hessenberg factor application
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

using namespace tlapack;

// This should really be moved to test utils or something
template <typename matrix_t>
inline void printMatrix(const matrix_t &A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i)
    {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(16) << A(i, j) << " ";
    }
}

TEST_CASE("left unmhr matches result from unghr", "[utils]")
{

    typedef std::complex<float> T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;
    using pair = std::pair<idx_t, idx_t>;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t n = 6;
    const idx_t ilo = 0;
    const idx_t ihi = 6;
    const real_t eps = uroundoff<real_t>();
    const bool verbose = false;

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _C(new T[n * n]);
    auto C = colmajor_matrix<T>(&_C[0], n, n);

    std::unique_ptr<T[]> _C_copy(new T[n * n]);
    auto C_copy = colmajor_matrix<T>(&_C_copy[0], n, n);

    std::unique_ptr<T[]> _C_copy2(new T[n * n]);
    auto C_copy2 = colmajor_matrix<T>(&_C_copy2[0], n, n);

    std::unique_ptr<T[]> _work(new T[2 * n]);
    auto work_m = colmajor_matrix<T>(&_work[0], n, 2);
    auto work = slice(work_m, pair{0, n}, 0);
    auto tau = slice(work_m, pair{0, n}, 1);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = zero;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = ihi + 1; i < n; ++i)
            A(i, j) = zero;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            C(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    lacpy(Uplo::General, C, C_copy);

    gehd2(ilo, ihi, A, tau, work);

    unmhr(Side::Left, Op::NoTrans, ilo, ihi, A, tau, C, work);

    if (verbose)
        printMatrix(C);

    unghr(ilo, ihi, A, tau, work);
    auto Q = slice(A, pair{ilo + 1, ihi}, pair{ilo + 1, ihi});
    auto C_slice = slice(C_copy, pair{ilo + 1, ihi}, pair{0, ncols(C)});
    auto C_slice2 = slice(C, pair{ilo + 1, ihi}, pair{0, ncols(C)});
    gemm(Op::NoTrans, Op::NoTrans, one, Q, C_slice, -one, C_slice2);

    if (verbose)
        printMatrix(C);

    real_t e_norm = lange( frob_norm, C_slice2 );

    CHECK(e_norm <= 1.0e2 * eps);
}

TEST_CASE("right unmhr matches result from unghr", "[utils]")
{

    typedef std::complex<float> T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;
    using pair = std::pair<idx_t, idx_t>;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t n = 6;
    const idx_t ilo = 1;
    const idx_t ihi = 5;
    const real_t eps = uroundoff<real_t>();
    const bool verbose = false;

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n);

    std::unique_ptr<T[]> _C(new T[n * n]);
    auto C = colmajor_matrix<T>(&_C[0], n, n);

    std::unique_ptr<T[]> _C_copy(new T[n * n]);
    auto C_copy = colmajor_matrix<T>(&_C_copy[0], n, n);

    std::unique_ptr<T[]> _C_copy2(new T[n * n]);
    auto C_copy2 = colmajor_matrix<T>(&_C_copy2[0], n, n);

    std::unique_ptr<T[]> _work(new T[2 * n]);
    auto work_m = colmajor_matrix<T>(&_work[0], n, 2);
    auto work = slice(work_m, pair{0, n}, 0);
    auto tau = slice(work_m, pair{0, n}, 1);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = zero;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = ihi + 1; i < n; ++i)
            A(i, j) = zero;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            C(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    lacpy(Uplo::General, C, C_copy);

    gehd2(ilo, ihi, A, tau, work);

    unmhr(Side::Right, Op::NoTrans, ilo, ihi, A, tau, C, work);

    if (verbose){
        std::cout<<std::endl<<"C";
        printMatrix(C);
    }

    unghr(ilo, ihi, A, tau, work);
    auto Q = slice(A, pair{ilo + 1, ihi}, pair{ilo + 1, ihi});
    auto C_slice = slice(C_copy, pair{0, nrows(C)}, pair{ilo+1, ihi});
    auto C_slice2 = slice(C, pair{0, nrows(C)}, pair{ilo+1, ihi});
    gemm(Op::NoTrans, Op::NoTrans, one, C_slice, Q, -one, C_slice2);

    if (verbose){
        std::cout<<std::endl<<"C";
        printMatrix(C);
    }

    real_t e_norm = lange( frob_norm, C_slice2 );

    CHECK(e_norm <= 1.0e2 * eps);
}