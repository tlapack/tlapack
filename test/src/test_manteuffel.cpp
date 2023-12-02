/// @file test_cauchy.cpp
/// @author Aslak Djupskås
/// @brief Test Manteuffel eigenvalues 
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/getri.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Manteuffel matrix properties",
                   "[manteuffel]", 
                   TLAPACK_TYPES_TO_TEST) 
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n = GENERATE(3);
    idx_t m = n * n;
    idx_t beta = GENERATE(1, 2);
    DYNAMIC_SECTION("n = " << n << " beta = " << beta) 
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(m * m) * eps;


        // Define the matrices and constants
        std::vector<T> M_;
        // Initialize matrix A which is n*n- by -n*n
        auto M = new_matrix(M_, n*n, n*n);
        
        // Initialize and compute C using the explicit formula
        // mm.generateManteuffel(A, n, 1, beta);
        mm.generateM_manteuffel(M, n);
        
        // calculate norm of C for later use in relative error
        real_t normA = tlapack::lange(tlapack::MAX_NORM, M);

    }
}