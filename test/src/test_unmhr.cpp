/// @file test_unmhr.cpp
/// @brief Test Hessenberg factor application
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("Result of unmhr matches result from unghr", "[eigenvalues][hessenberg]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using pair = std::pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    auto matrix_type = GENERATE(as<std::string>{}, "Random");
    Side side = GENERATE(Side::Left, Side::Right);
    Op op = GENERATE(Op::NoTrans, Op::ConjTrans);

    idx_t m = 12;
    idx_t n = 10;
    idx_t ilo = GENERATE(0, 1);
    idx_t ihi = GENERATE(9, 10);

    const T zero(0);
    const T one(1);
    const real_type<T> eps = uroundoff<real_type<T>>();
    const real_type<T> tol = n * 1.0e2 * eps;

    // Define the matrices and vectors
    std::unique_ptr<T[]> H_(new T[n * n]);
    std::unique_ptr<T[]> C_(new T[m * n]);
    std::unique_ptr<T[]> C_copy_(new T[m * n]);

    // This only works for legacy matrix, we really work on that construct_matrix function
    auto H = new_matrix( &H_[0], n, n );
    auto C = new_matrix( &C_[0], m, n );
    auto C_copy = new_matrix( &C_copy_[0], m, n );
    std::vector<T> tau(n);
    std::vector<T> work(std::max(n,m));

    if (matrix_type == "Random")
    {
        // Generate a random matrix in H
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                H(i, j) = rand_helper<T>();

        // Generate a random matrix in C
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                C(i, j) = rand_helper<T>();
    }
    lacpy(Uplo::General, C, C_copy);

    // Make sure ilo and ihi correspond to the actual matrix
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            H(i, j) = zero;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i; ++j)
            H(i, j) = zero;

    // Hessenberg reduction of H
    gehd2(ilo, ihi, H, tau, work);

    DYNAMIC_SECTION("UNMHR with"
                    << " matrix = " << matrix_type << " ilo = " << ilo << " ihi = " << ihi)
    {

        real_t c_norm = lange(frob_norm, C);

        // Apply the orthogonal factor to C
        unmhr(side, op, ilo, ihi, H, tau, C, work);

        // Generate the orthogonal factor
        unghr(ilo, ihi, H, tau, work);

        // Multiply C_copy with the orthogonal factor
        auto Q = slice(H, pair{ilo + 1, ihi}, pair{ilo + 1, ihi});
        if (side == Side::Left)
        {
            auto C_copy_s = slice(C_copy, pair{ilo + 1, ihi}, pair{0, ncols(C)});
            auto C_s = slice(C, pair{ilo + 1, ihi}, pair{0, ncols(C)});
            gemm(op, Op::NoTrans, one, Q, C_copy_s, -one, C_s);
            for (idx_t i = 0; i < ilo+1; ++i)
                for (idx_t j = 0; j < ncols(C); ++j)
                    C(i, j) =  C(i,j) - C_copy(i,j);
            for (idx_t i = ihi; i < nrows(C); ++i)
                for (idx_t j = 0; j < ncols(C); ++j)
                    C(i, j) =  C(i,j) - C_copy(i,j);
        }
        else
        {
            auto C_copy_s = slice(C_copy, pair{0, nrows(C)}, pair{ilo + 1, ihi});
            auto C_s = slice(C, pair{0, nrows(C)}, pair{ilo + 1, ihi});
            gemm(Op::NoTrans, op, one, C_copy_s, Q, -one, C_s);
            for (idx_t j = 0; j < ilo+1; ++j)
                for (idx_t i = 0; i < nrows(C); ++i)
                    C(i, j) =  C(i,j) - C_copy(i,j);
            for (idx_t j = ihi; j < ncols(C); ++j)
                for (idx_t i = 0; i < nrows(C); ++i)
                    C(i, j) =  C(i,j) - C_copy(i,j);
        }

        real_t e_norm = lange(frob_norm, C);

        CHECK(e_norm <= tol * c_norm);
    }
}
