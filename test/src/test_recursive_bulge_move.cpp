/// @file test_gehrd.cpp
/// @brief Test hessenberg reduction
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
#include <plugins/tlapack_debugutils.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("Non recursive moves bulges is backward stable", "[eigenvalues][hessenberg]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using pair = std::pair<idx_t, idx_t>;

    rand_generator gen;

    // Number of shifts
    idx_t ns = GENERATE(2, 4, 6);
    // Number of positions to move the shifts
    idx_t np = GENERATE(1, 2, 4);
    // size of matrix
    idx_t n = ns + 1 + np;
    // Number of bulges
    idx_t nb = ns / 2;

    const real_t eps = uroundoff<real_t>();
    const real_t tol = 1.0e2 * n * eps;

    // Define the matrices and vectors
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> V_(new T[3 * nb]);
    std::unique_ptr<T[]> A_copy_(new T[n * n]);
    std::unique_ptr<T[]> Q_(new T[n * n]);
    std::unique_ptr<T[]> Q1_(new T[n * n]);
    std::unique_ptr<T[]> Q2_(new T[(n - 1) * (n - 1)]);

    // This only works for legacy matrix, we really need to work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(n, n, &A_copy_[0], n);
    auto Q = legacyMatrix<T, layout<matrix_t>>(n, n, &Q_[0], n);
    auto Q1 = legacyMatrix<T, layout<matrix_t>>(n, n, &Q1_[0], n);
    auto Q2 = legacyMatrix<T, layout<matrix_t>>(n - 1, n - 1, &Q2_[0], n - 1);
    auto V = legacyMatrix<T, layout<matrix_t>>(3, nb, &V_[0], layout<matrix_t> == Layout ::ColMajor ? 3 : nb);
    std::vector<std::complex<real_t>> s(ns);

    // Generate a random Hessenberg matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < min(j + 2, n); ++i)
            A(i, j) = rand_helper<T>(gen);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            A(i, j) = (T)0.0;

    tlapack::lacpy(Uplo::General, A, A_copy);

    // Pick some shifts to use
    for (idx_t i = 0; i < ns; i = i + 2)
    {
        s[i] = std::complex<real_t>((real_t)i, (real_t) + 1);
        s[i + 1] = std::complex<real_t>((real_t)i, (real_t)-1);
    }

    introduce_bulges(A, s, Q1, V);

    // laset(Uplo::General, (T) 0.0, (T) 1.0, Q2);
    move_bulges(A, s, Q2, V);

    // Merge Q1 and Q2 into Q
    lacpy(Uplo::General, Q1, Q);
    auto Q1_slice = slice(Q1, pair{0, n}, pair{1, n});
    auto Q_slice = slice(Q, pair{0, n}, pair{1, n});
    gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, Q1_slice, Q2, (real_t)0.0, Q_slice);

    remove_bulges(A, s, Q2, V);

    lacpy(Uplo::General, Q, Q1);
    gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, Q1_slice, Q2, (real_t)0.0, Q_slice);

    std::unique_ptr<T[]> _res_orth(new T[n * n]);
    auto res_orth = legacyMatrix<T, layout<matrix_t>>(n, n, &_res_orth[0], n);
    auto orth_res_norm = check_orthogonality(Q, res_orth);
    CHECK(orth_res_norm <= tol);

    std::unique_ptr<T[]> _Q2(new T[n * n]);
    std::unique_ptr<T[]> _res(new T[n * n]);
    std::unique_ptr<T[]> _work(new T[n * n]);
    auto res = legacyMatrix<T, layout<matrix_t>>(n, n, &_res[0], n);
    auto work = legacyMatrix<T, layout<matrix_t>>(n, n, &_work[0], n);
    auto normA = tlapack::lange(tlapack::frob_norm, A_copy);

    gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
    gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)0.0, res);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            res(i, j) -= A(i, j);
    auto simil_res_norm = tlapack::lange(tlapack::frob_norm, res);
    CHECK(simil_res_norm <= tol * normA);
}

TEMPLATE_LIST_TEST_CASE("Recursive move bulges is backward stable", "[eigenvalues][hessenberg]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using pair = std::pair<idx_t, idx_t>;

    rand_generator gen;

    // Number of shifts
    idx_t ns = GENERATE(2,4,6,10);
    // Number of positions to move the shifts
    idx_t np = GENERATE(2,4,6,10);
    // Optimization parameter
    idx_t nx = GENERATE(2, 4);
    // size of matrix
    idx_t n = ns + 1 + np;
    // Number of bulges
    idx_t nb = ns / 2;

    const real_t eps = uroundoff<real_t>();
    const real_t tol = 1.0e2 * n * eps;

    // Define the matrices and vectors
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> V_(new T[3 * nb]);
    std::unique_ptr<T[]> A_copy_(new T[n * n]);
    std::unique_ptr<T[]> Q_(new T[n * n]);
    std::unique_ptr<T[]> Q1_(new T[n * n]);
    std::unique_ptr<T[]> Q2_(new T[(n - 1) * (n - 1)]);

    // This only works for legacy matrix, we really need to work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(n, n, &A_copy_[0], n);
    auto Q = legacyMatrix<T, layout<matrix_t>>(n, n, &Q_[0], n);
    auto Q1 = legacyMatrix<T, layout<matrix_t>>(n, n, &Q1_[0], n);
    auto Q2 = legacyMatrix<T, layout<matrix_t>>(n - 1, n - 1, &Q2_[0], n - 1);
    auto V = legacyMatrix<T, layout<matrix_t>>(3, nb, &V_[0], layout<matrix_t> == Layout ::ColMajor ? 3 : nb);
    std::vector<std::complex<real_t>> s(ns);

    // Generate a random Hessenberg matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < min(j + 2, n); ++i)
            A(i, j) = rand_helper<T>(gen);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            A(i, j) = (T)0.0;

    tlapack::lacpy(Uplo::General, A, A_copy);

    // Pick some shifts to use
    for (idx_t i = 0; i < ns; i = i + 2)
    {
        s[i] = std::complex<real_t>((real_t)i, (real_t) + 1);
        s[i + 1] = std::complex<real_t>((real_t)i, (real_t)-1);
    }

    DYNAMIC_SECTION("Recursive bulge move with"
                    << " ns = " << ns << " np = " << np << " nx = " << nx)
    {

        introduce_bulges(A, s, Q1, V);

        std::cout<<"A"<<std::endl;
        print_matrix( A );

        move_bulges_opts_t<idx_t, T> opts;
        opts.nx = nx;
        move_bulges_recursive(A, s, Q2, V);

        std::cout<<"A"<<std::endl;
        print_matrix( A );

        // Merge Q1 and Q2 into Q
        lacpy(Uplo::General, Q1, Q);
        auto Q1_slice = slice(Q1, pair{0, n}, pair{1, n});
        auto Q_slice = slice(Q, pair{0, n}, pair{1, n});
        gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, Q1_slice, Q2, (real_t)0.0, Q_slice);

        remove_bulges(A, s, Q2, V);
        lacpy(Uplo::General, Q, Q1);
        gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, Q1_slice, Q2, (real_t)0.0, Q_slice);

        std::cout<<"A"<<std::endl;
        print_matrix( A );

        std::unique_ptr<T[]> _res_orth(new T[n * n]);
        auto res_orth = legacyMatrix<T, layout<matrix_t>>(n, n, &_res_orth[0], n);
        auto orth_res_norm = check_orthogonality(Q, res_orth);
        CHECK(orth_res_norm <= tol);

        std::unique_ptr<T[]> _Q2(new T[n * n]);
        std::unique_ptr<T[]> _res(new T[n * n]);
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto res = legacyMatrix<T, layout<matrix_t>>(n, n, &_res[0], n);
        auto work = legacyMatrix<T, layout<matrix_t>>(n, n, &_work[0], n);
        auto normA = tlapack::lange(tlapack::frob_norm, A_copy);

        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A_copy, (T)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)0.0, res);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                res(i, j) -= A(i, j);

        std::cout<<"A"<<std::endl;
        print_matrix( A );

        auto simil_res_norm = tlapack::lange(tlapack::frob_norm, res);
        CHECK(simil_res_norm <= tol * normA);
    }
}
