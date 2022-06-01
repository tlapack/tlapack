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

TEMPLATE_LIST_TEST_CASE("Moves bulges is backward stable (or as stable as can be)", "[eigenvalues][hessenberg]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using pair = std::pair<idx_t,idx_t>;

    rand_generator gen;

    // Number of shifts
    idx_t ns = GENERATE(2, 4, 6);
    // Number of positions to move the shifts
    idx_t np = GENERATE(1);
    // size of matrix
    idx_t n = ns + 2 + np;
    // Number of bulges
    idx_t nb = ns / 2;

    // Define the matrices and vectors
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> V_(new T[3 * nb]);
    std::unique_ptr<T[]> A_copy_(new T[n * n]);
    std::unique_ptr<T[]> Q_(new T[(n - 2) * (n - 2)]);

    // This only works for legacy matrix, we really need to work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(n, n, &A_copy_[0], n);
    auto Q = legacyMatrix<T, layout<matrix_t>>(n - 2, n - 2, &Q_[0], n - 2);
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

    // Introduce some bulges into the matrix
    for (idx_t i2 = nb; i2 > 0; --i2)
    {
        // Generate random householder reflector
        T tau;
        idx_t i = i2 - 1;
        auto v = col(V, i);
        v[0] = rand_helper<T>(gen);
        v[1] = rand_helper<T>(gen);
        v[2] = rand_helper<T>(gen);
        larfg(v, tau);
        v[0] = tau;

        // Partially apply reflector to A
        idx_t i_pos = 2 * i;
        auto t0 = v[0];
        auto t1 = t0 * conj(v[1]);
        auto t2 = t0 * conj(v[2]);
        for (idx_t j = 0; j < i_pos + 3; ++j)
        {
            auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
            A(j, i_pos) = A(j, i_pos) - sum * t0;
            A(j, i_pos + 1) = A(j, i_pos + 1) - sum * t1;
            A(j, i_pos + 2) = A(j, i_pos + 2) - sum * t2;
        }

        // Fully apply reflector to A_copy
        for (idx_t j = 0; j < n; ++j)
        {
            auto sum = A_copy(j, i_pos) + v[1] * A_copy(j, i_pos + 1) + v[2] * A_copy(j, i_pos + 2);
            A_copy(j, i_pos) = A_copy(j, i_pos) - sum * t0;
            A_copy(j, i_pos + 1) = A_copy(j, i_pos + 1) - sum * t1;
            A_copy(j, i_pos + 2) = A_copy(j, i_pos + 2) - sum * t2;
        }
    }

    move_bulges(A, s, Q, V);

    std::unique_ptr<T[]> _res_orth(new T[(n - 2) * (n - 2)]);
    auto res_orth = legacyMatrix<T, layout<matrix_t>>(n-2, n-2, &_res_orth[0], n-2);

    auto orth_res_norm = check_orthogonality(Q, res_orth);
        // CHECK(orth_res_norm <= tol);

    std::unique_ptr<T[]> _Q2(new T[n * n]);
    std::unique_ptr<T[]> _res(new T[n * n]);
    std::unique_ptr<T[]> _work(new T[n * n]);
    auto Q2 = legacyMatrix<T, layout<matrix_t>>(n, n, &_Q2[0], n);
    auto res = legacyMatrix<T, layout<matrix_t>>(n, n, &_res[0], n);
    auto work = legacyMatrix<T, layout<matrix_t>>(n, n, &_work[0], n);
    auto normA = tlapack::lange(tlapack::frob_norm, A_copy);

    // Copy Q into larger Q2 so it is easier to work with.
    laset( Uplo::General, (T)0.0, (T)1.0, Q2 );
    auto Q2_slice = slice( Q2, pair{1,n-1}, pair{1,n-1});
    lacpy( Uplo::General, Q, Q2_slice);

    laset( Uplo::General, (T)0.0, (T)0.0, res );
    laset( Uplo::General, (T)0.0, (T)0.0, work );
    
    gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q2, A_copy, (T)0.0, work);
    gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q2, (T)0.0, res);

    // Undo the partial application of the reflectors in A so we can check for errors
    // for (idx_t i = 0; i < nb; ++i)
    // {
    //     auto v = col(V, i);
    //     idx_t i_pos = 2 * i + np;
    //     auto t0 = conj(v[0]);
    //     auto t1 = t0 * conj(v[1]);
    //     auto t2 = t0 * conj(v[2]);
    //     for (idx_t j = 0; j < i_pos + 3; ++j)
    //     {
    //         auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
    //         A(j, i_pos) = A(j, i_pos) - sum * t0;
    //         A(j, i_pos + 1) = A(j, i_pos + 1) - sum * t1;
    //         A(j, i_pos + 2) = A(j, i_pos + 2) - sum * t2;
    //     }
    // }
    
    // // Undo the full application of the reflectors in res so we can check for errors
    // for (idx_t i = 0; i < nb; ++i)
    // {
    //     auto v = col(V, i);
    //     idx_t i_pos = 2 * i + np;
    //     auto t0 = conj(v[0]);
    //     auto t1 = t0 * conj(v[1]);
    //     auto t2 = t0 * conj(v[2]);
    //     for (idx_t j = 0; j < n; ++j)
    //     {
    //         auto sum = res(j, i_pos) + v[1] * res(j, i_pos + 1) + v[2] * res(j, i_pos + 2);
    //         res(j, i_pos) = res(j, i_pos) - sum * t0;
    //         res(j, i_pos + 1) = res(j, i_pos + 1) - sum * t1;
    //         res(j, i_pos + 2) = res(j, i_pos + 2) - sum * t2;
    //     }
    // }

    // for(idx_t j = 0; j < n; ++j)
    //     for(idx_t i =0;i<n;++i)
    //         res(i,j) -= A(i,j);

    int t = 2;
}
