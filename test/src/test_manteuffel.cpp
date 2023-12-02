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
#include <vector>
#include <cmath>

// Define pi
const double pi = std::acos(-1);

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

    idx_t n = GENERATE(3, 5);
    idx_t m = n * n;
    real_t beta = GENERATE(1, 2);
    real_t h = 1.0;
    real_t L = h*(n+1);


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
        
        // Define the matrices and constants
        std::vector<T> N_;
        // Initialize matrix A which is n*n- by -n*n
        auto N = new_matrix(N_, n*n, n*n);
        
        // Define the matrices and constants
        std::vector<T> A_;
        // Initialize matrix A which is n*n- by -n*n
        auto A = new_matrix(A_, n*n, n*n);
        

        // Initialize and compute C using the explicit formula
        // mm.generateManteuffel(A, n, 1, beta);
        mm.generateM_manteuffel(M, n);
        mm.generateN_manteuffel(N, n);
        mm.generateManteuffel(A, M, N, n, h, beta);

 
        // compute eigenvalues of A
        std::vector<T> evals(n*n);
        idx_t i = 0;

        for (idx_t k = 0; k < n; ++k) {
            for (idx_t j = 0; j < n; ++j) {
                double cos_j = std::cos((j+1) * pi / L);
                double cos_k = std::cos((k+1) * pi / L);
                double term = std::sqrt(1.0 - std::pow(beta / 2.0, 2.0)) * (cos_j + cos_k);
                evals[i] = 2 * (2 - term);
                i++;
            }
        }

       // calculate norm of C for later use in relative error
        real_t normA = tlapack::lange(tlapack::MAX_NORM, A);
        

    }
}