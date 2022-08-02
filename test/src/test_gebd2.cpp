/// @file test_gebd2.cpp
/// @brief Test GEBD2 using UNG2R and UNGL2. Output an upper bidiagonal matrix B for a m-by-n matrix A (m >= n).
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("bidiagonal reduction is backward stable", "[bidiagonal][svd]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    const T zero(0);
    const T one(1);

    idx_t m, n;

    m = GENERATE(10, 15, 20, 30);
    n = GENERATE(10, 12, 20, 30);

    if (m >= n) // Only m >= n matrices are supported (yet). gebd2 will give upper bidiagonal matrix B
    {

        const real_t eps = ulp<real_t>();
        const real_t tol = max(m, n) * eps;

        std::unique_ptr<T[]> A_(new T[m * n]);
        std::unique_ptr<T[]> A_copy_(new T[m * n]);

        auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
        auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n);

        std::vector<T> work(m); // max of m and n
        std::vector<T> tauv(n); // min of m and n
        std::vector<T> tauw(n); // min of m and n

        // Generate random m-by-n matrix
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        lacpy(Uplo::General, A, A_copy);

        DYNAMIC_SECTION("m = " << m << " n = " << n)
        {
            gebd2(A, tauv, tauw, work);

            // Get upper bidiagonal B
            std::unique_ptr<T[]> B_(new T[m * n]);
            auto B = legacyMatrix<T, layout<matrix_t>>(m, n, &B_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
            laset(Uplo::General, zero, zero, B);

            B(0, 0) = A(0, 0);
            for (idx_t j = 1; j < n; ++j)
            {
                B(j - 1, j) = A(j - 1, j);
                B(j, j) = A(j, j);
            }
            real_t normB = lange(Norm::Max, B);

            // Generate unitary matrix Q of m-by-m
            std::unique_ptr<T[]> Q_(new T[m * m]);
            auto Q = legacyMatrix<T, layout<matrix_t>>(m, m, &Q_[0], m);
            lacpy(Uplo::Lower, A, Q);

            ung2r(n, Q, tauv, work);

            // Test for Q's orthogonality
            std::unique_ptr<T[]> _Wq(new T[m * m]);
            auto Wq = legacyMatrix<T, layout<matrix_t>>(m, m, &_Wq[0], m);
            auto orth_Q = check_orthogonality(Q, Wq);
            CHECK(orth_Q <= tol);

            // Generate unitary matrix Z of n-by-n
            std::unique_ptr<T[]> Z_(new T[n * n]);
            auto Z = legacyMatrix<T, layout<matrix_t>>(n, n, &Z_[0], n);
            laset(Uplo::General, zero, one, Z); // Initialize Z as identity matrix.

            // Slice Z down to Z11 of size (n-1)-by-(n-1) and copy upper A to Z11
            auto X = slice(A, range(0,n-1), range(1,n)); // X is (n-1)-by-(n-1) slice of A
            auto Z11 = slice(Z, range(1, n), range(1, n));
            lacpy(Uplo::General, X, Z11);

            ungl2(Z11, tauw, work); // Note: the unitary matrix Z we get here is ConjTransed

            // Test for Z's orthogonality
            std::unique_ptr<T[]> _Wz(new T[n * n]);
            auto Wz = legacyMatrix<T, layout<matrix_t>>(n, n, &_Wz[0], n);
            laset(Uplo::General, zero, one, Wz);
            auto orth_Z = check_orthogonality(Z, Wz);
            CHECK(orth_Z <= tol);

            // Test B = Q_H * A * Z
            // Generate a zero matrix K of size m-by-n to be the product of Q_H * A
            std::unique_ptr<T[]> K_(new T[m * n]);
            auto K = legacyMatrix<T, layout<matrix_t>>(m, n, &K_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
            laset(Uplo::General, zero, zero, K);
            gemm(Op::ConjTrans, Op::NoTrans, real_t(1.), Q, A_copy, real_t(0), K);

            // B = K * Z - B
            gemm(Op::NoTrans, Op::ConjTrans, real_t(1.), K, Z, real_t(-1.), B);

            real_t repres = lange(Norm::Max, B);
            CHECK(repres <= tol * normB);
            
        }
    }
}