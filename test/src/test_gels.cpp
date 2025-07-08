/// @file test_bidiag.cpp
/// @author David Li, University of California, Berkeley, USA
/// @brief Test gels reduction
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

#include <tlapack/lapack/gels.hpp>
#include <tlapack/blas/asum.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("gels reduction is backward stable",
                   "[gels]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t m = GENERATE(10, 20, 50);
    const idx_t n = GENERATE(10);
    
    // Solve the least squares problem

    // Check correctness

    DYNAMIC_SECTION("m = " << m << "n = " << n)
    {
       
        Op trans = Op::NoTrans;
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        idx_t k = min<idx_t>(m, n);
        // Create matrices
        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> A__;
        auto A_copy = new_matrix(A__, m, n);
           mm.random(A);
        lacpy(GENERAL, A, A_copy);  // Copy A to A_copy

        std::vector<T> B_;
        auto B = new_matrix(B_, m, n); // Potential TODO to check size
        std::vector<T> B__;
        auto B_copy = new_matrix(B__, m, n);
           mm.random(B);
        lacpy(GENERAL,B, B_copy);  // Copy B to B_copy

        // Calculate with gels
        int info = gels(A, B, trans); // B is the output X
        REQUIRE(info == 0);

        gemm(Op::NoTrans, Op::NoTrans, T(-1), A_copy, B, T(1), B_copy); // B_copy = B_copy - A_copy * B
   /*
        // Calculate the norm of A
        real_t normA = lange(Norm::One, A_copy);

        // Calculate the norm of B
        real_t normB = lange(Norm::One, B_copy);
        
        gemm(Op::NoTrans, Op::NoTrans, T(-1), A_copy, B, T(1), B_copy); // B_copy = B_copy - A_copy * B

        // Calculate the norm of the residual
        real_t res = lange(Norm::One, B_copy);

        // Calculate the norm of X
        real_t normX = lange(Norm::One, B);

        // Calculate the norm of the residual
        real_t normR = res / (max(m,n)*normA * normX*eps);
        */
        // Calculate the norm of A
        real_t normA = lange(Norm::One, A_copy);
        real_t res = 0;
        for (real_t i = 0; i < n; i++)
        {
            real_t bnorm = asum(slice(B_copy, range(0, m), i));
            real_t xnorm = asum(slice(B, range(0, n), i));
            res = max(res, ((bnorm / normA)/ xnorm)/(max(m,n)*eps));
        }
    }

}