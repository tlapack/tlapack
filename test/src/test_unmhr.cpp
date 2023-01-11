/// @file test_unmhr.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test Hessenberg factor application
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/lapack/unghr.hpp>
#include <tlapack/lapack/unmhr.hpp>
#include <tlapack/lapack/gehd2.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Result of unmhr matches result from unghr", "[eigenvalues][hessenberg]", TLAPACK_TYPES_TO_TEST)
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

    INFO( "matrix_type = " << matrix_type );
    INFO( "side = " << side );
    INFO( "Op = " << op );

    idx_t m = 12;
    idx_t n = 10;
    idx_t ilo = GENERATE(0, 1);
    idx_t ihi = GENERATE(9, 10);

    INFO("ilo = " << ilo);
    INFO("ihi = " << ihi);

    const T zero(0);
    const T one(1);
    const real_type<T> eps = uroundoff<real_type<T>>();
    const real_type<T> tol = real_t(n * 1.0e2) * eps;

    // Define the matrices
    std::vector<T> H_; auto H = new_matrix( H_, n, n );
    std::vector<T> C_; auto C = new_matrix( C_, m, n );
    std::vector<T> C_copy_; auto C_copy = new_matrix( C_copy_, m, n );
    std::vector<T> tau(n);

    // Workspace computation:
    workinfo_t workinfo = {};
    gehd2_worksize(ilo, ihi, H, tau, workinfo);
    unmhr_worksize(side, op, ilo, ihi, H, tau, C, workinfo);
    unghr_worksize(ilo, ihi, H, tau, workinfo);

    // Workspace allocation:
    vectorOfBytes workVec;
    workspace_opts_t<> workOpts( alloc_workspace( workVec, workinfo ) );

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
    gehd2(ilo, ihi, H, tau, workOpts);

    {

        real_t c_norm = lange(frob_norm, C);

        // Apply the orthogonal factor to C
        unmhr(side, op, ilo, ihi, H, tau, C, workOpts);

        // Generate the orthogonal factor
        unghr(ilo, ihi, H, tau, workOpts);

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
