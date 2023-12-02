/// @file test_cauchy.cpp
/// @author Aslak Djupskås
/// @brief Test Cauchy invertion 
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

TEMPLATE_TEST_CASE("Cauchy matrix properties",
                   "[getri][cauchy]", 
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

    idx_t n = GENERATE(3, 9);
    GetriVariant variant = GENERATE(GetriVariant::UXLI, GetriVariant::UILI);

    DYNAMIC_SECTION("n = " << n << " variant = " << (char)variant) 
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n * n) * eps;

        // Define the matrices and vectors
        std::vector<T> C_;
        auto C = new_matrix(C_, n, n);
        
        std::vector<T> invCexpl_;
        auto invCexpl = new_matrix(invCexpl_, n, n);
        
        std::vector<T> invC_;
        auto invC = new_matrix(invC_, n, n);
        
        std::vector<T> x(n);
        std::vector<T> y(n);

        for (idx_t i = 0; i < n; ++i)
        {
            x[i] = (T)(i+1);
            y[i] = (T)((i+1+n));
        }

        // Initialize and compute C using the explicit formula
        mm.generateCauchy(C, x, y); 
        mm.generateInverseCauchy(invCexpl, x, y);
  
        // make a deep copy A
        lacpy(GENERAL, C, invC);


        // calculate norm of C for later use in relative error
        real_t normC = tlapack::lange(tlapack::MAX_NORM, C);

        // building error matrix E 
        std::vector<T> E1_;
        auto E1 = new_matrix(E1_, n, n);
        
        // E <----- inv(C)*C - I
        gemm(NO_TRANS, NO_TRANS, real_t(1), C, invCexpl, E1);
        for (idx_t i = 0; i < n; i++)
            E1(i, i) -= real_t(1);

        // error is  || inv(C)*C - I || / ( ||C|| * ||inv(C)|| )
        real_t error1 = tlapack::lange(tlapack::MAX_NORM, E1) /
                       (normC * tlapack::lange(tlapack::MAX_NORM, invCexpl));

        UNSCOPED_INFO("|| inv(C)*C - I || / ( ||C|| * ||inv(C)|| )");
        CHECK(error1 / tol <= real_t(1));  // tests if error<=tol

        // building error matrix E2
        std::vector<T> E2_;
        auto E2 = new_matrix(E2_, n, n);

        // LU factorize Pivoted C
        std::vector<idx_t> piv(n, idx_t(0));
        getrf(invC, piv);

        // run inverse function, this could test any inverse function of choice
        GetriOpts opts;
        opts.variant = variant;
        getri(invC, piv, opts);

        // E <----- inv(C)*C - I
        gemm(NO_TRANS, NO_TRANS, real_t(1), C, invC, E2);
        for (idx_t i = 0; i < n; i++)
            E2(i, i) -= real_t(1);

        
        // error is  || inv(C)*C - I || / ( ||C|| * ||inv(C)|| )
        real_t error2 = tlapack::lange(tlapack::MAX_NORM, E2) /
                       (normC * tlapack::lange(tlapack::MAX_NORM, invC));

        UNSCOPED_INFO("|| inv(C)*C - I || / ( ||C|| * ||inv(C)|| )");
        CHECK(error2 / tol <= real_t(1));  // tests if error<=tol

        
        // building error matrix E3
        std::vector<T> E3_;
        auto E3 = new_matrix(E3_, n, n);
        for (idx_t i = 0; i < n; i++) 
            for(idx_t j = 0; j < n; j++) 
                E3(i, j) = invC(i, j) - invCexpl(i, j);
            
        // error is  || inv(C)*C - I || / ( ||C|| * ||inv(C)|| )
        real_t error3 = tlapack::lange(tlapack::MAX_NORM, E3) /
                       (tlapack::lange(tlapack::MAX_NORM, invCexpl) * tlapack::lange(tlapack::MAX_NORM, invC));

        UNSCOPED_INFO("|| inv(C)*C - I || / ( ||C|| * ||inv(C)|| )");
        CHECK(error3 / tol <= real_t(1));  // tests if error<=tol
    }
}

