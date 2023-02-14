/// @file test_hetd2.cpp
/// @author Skylar Johns, University of Colorado Denver, USA
/// @brief Test HETD2
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/plugins/debugutils.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/hetd2.hpp>

// This will need to be removed
#include <tlapack/lapack/gehd2.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/unghr.hpp>


using namespace tlapack;

TEMPLATE_TEST_CASE("Tridiagnolization of a symmetric matrix works",
                   "[hetd2]",
                    //legacyMatrix<float>)
                    legacyMatrix<std::complex<float>>)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const real_t zero(0);

    const real_t one(1);

    idx_t n;

    n = GENERATE(6);
    const Uplo uplo = GENERATE(Uplo::Lower);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(n * n) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<real_t> E(n-1), D(n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>();

    real_t normA = lange(Norm::Fro, A);

    lacpy(uplo, A, Q);

// Think about this, we need tau of size n for unghr
// But maybe n-1 is better
//    std::vector<T> tauw(n - 1);
        std::vector<T> tauw(n);

//    hetd2(uplo, Q, tauw);

    for (idx_t j = 0; j < n; ++j) {
        Q(j, j) = real(Q(j,j));
        for (idx_t i = j+1; i < n; ++i)
            Q(j, i) = conj(Q(i,j));
    }


    for (idx_t i = 0; i < n; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(3) << Q(i, j) << " ";
            //std::printf("%+8.4f",Q(i,j));
    }
    std::cout << std::endl;

    gehd2(0,n, Q, tauw);

    for (idx_t i = 0; i < n; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(3) << Q(i, j) << " ";
            //std::printf("%+8.4f",Q(i,j));
    }
    std::cout << std::endl;

    for (idx_t i = 0; i < n; ++i)
        D[i] = real(Q(i,i));

    for (idx_t i = 0; i < n-1; ++i)
        E[i] = real(Q(i+1,i));


    std::cout << std::endl;

    for (idx_t i = 0; i < n; ++i)
        std::cout << std::setw(3) << D[i] << " ";
    std::cout << std::endl;

    for (idx_t i = 0; i < n-1; ++i)
        std::cout << std::setw(3) << E[i] << " ";
    std::cout << std::endl;

    /// TODO: Review the following code

    //ung2r(n - 1, Q, tauw);
    
    unghr(0,n, Q, tauw);


    auto orth_Q = check_orthogonality(Q);
    CHECK(orth_Q <= tol);




    // // When upper and positive, e = 1
    // if (uplo == Uplo::Upper) {
    //     auto E = diag(A, 1);
    // }
    // // Else, e = -1
    // else {
    //     auto E = diag(A, -1);
    // }
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> Z_;
    auto Z = new_matrix(Z_, n, n);
    // Compute R = QB
    for (idx_t i = 0; i != n; ++i) {
        R(i, 0) = Q(i, 0) * D[0] + Q(i, 1) * E[0];
        for (idx_t j = 1; j != n - 1; ++j) {
            R(i, j) = Q(i, j) * D[j] + Q(i, j + 1) * E[j] + Q(i, j) * conj(E[j - 1]);
        }
        R(i, n - 1) = Q(i, n - 1) * D[n - 1] + Q(i, n - 1) * conj(E[n - 2]);
    }
    
    gemm(Op::NoTrans, Op::ConjTrans, one, R, Q, -one, A);

    CHECK(lange(Norm::Fro, A) / normA < tol);

    // lacpy(uplo, slice(A, range(0, min(m, k)), range(0, n)), Q);

    // // Workspace computation:
    // workinfo_t workinfo = {};
    // gelq2_worksize(A, tauw, workinfo);
    // ungl2_worksize(Q, tauw, workinfo);

    // // Workspace allocation:
    // vectorOfBytes workVec;
    // workspace_opts_t<> workOpts(alloc_workspace(workVec, workinfo));

    // for (idx_t j = 0; j < n; ++j)
    //     for (idx_t i = 0; i < m; ++i)
    //         A(i, j) = rand_helper<T>();

    // lacpy(Uplo::General, A, A_copy);

    // if (k <= n)  // k must be less than or equal to n, because we cannot get
    // a Q
    //              // bigger than n-by-n
    // {
    //     INFO("m = " << m << " n = " << n << " k = " << k);
    //     {
    //         gelq2(A, tauw, workOpts);

    //         // Q is sliced down to the desired size of output Q (k-by-n).
    //         // It stores the desired number of Householder reflectors that
    //         UNGL2
    //         // will use.
    //         lacpy(Uplo::General, slice(A, range(0, min(m, k)), range(0, n)),
    //         Q);

    //         ungl2(Q, tauw, workOpts);

    //         // Wq is the identity matrix to check the orthogonality of Q
    //         std::vector<T> Wq_;
    //         auto Wq = new_matrix(Wq_, k, k);
    //         auto orth_Q = check_orthogonality(Q, Wq);
    //         CHECK(orth_Q <= tol);

    //         // L is sliced from A after GELQ2
    //         std::vector<T> L_;
    //         auto L = new_matrix(L_, min(k, m), k);
    //         laset(Uplo::Upper, zero, zero, L);
    //         lacpy(Uplo::Lower, slice(A, range(0, min(m, k)), range(0, k)),
    //         L);

    //         // R stores the product of L and Q
    //         std::vector<T> R_;
    //         auto R = new_matrix(R_, min(k, m), n);

    //         // Test A = L * Q
    //         gemm(Op::NoTrans, Op::NoTrans, real_t(1.), L, Q, R);
    //         for (idx_t j = 0; j < n; ++j)
    //             for (idx_t i = 0; i < min(m, k); ++i)
    //                 A_copy(i, j) -= R(i, j);

    //         real_t repres = lange(
    //             Norm::Max, slice(A_copy, range(0, min(m, k)), range(0, n)));
    //         CHECK(repres <= tol);
    //     }
    // }
}
